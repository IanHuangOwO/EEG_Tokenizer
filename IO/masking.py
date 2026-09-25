"""
Pretrain masking (masked phase only; the tokenizer phase ignores masks).

One interface for every strategy, so the training loop never needs to know which one runs:

    strategy.set_epoch(e)     # e = masked-phase epoch, 1-based; the strategy owns its schedule
    strategy.state()          # hashable; the dataset redraws its masks when this changes
    strategy.multiplier       # dataset copies per epoch (2 = every window shown twice)
    strategy.generate(C, N, valid, coords) -> [multiplier, C*N] bool, one mask per copy
    strategy.describe()       # one line for the log

Tokens are channel-major (c * N + n), matching IO/dataset.py's (C, N) layout. valid (bool,
same shape) marks real content: masks are never True on a zero-padded channel or on an
assembled window's zero tail, and every ratio is a ratio of the VALID tokens -- otherwise a
heavily padded dataset (8 of 64 channels) or a window's zero tail wastes most of the budget
on content already known to be zero (docs/model-analysis-checklist.md).

Strategies (preprocess_params.mask.masking_strategy):
  random                   token masking at a fixed ratio
  complementary            fixed 0.5 random mask, each window shown with its inverse too
  random_to_complementary  random ratio ramp, then complementary (the pre-2026-09-25 default)
  mixture                  one MaskMode per window (channel cluster / random channel / time
                           block, ...), all on one shared ramp -- the modular one; new mask
                           patterns are new MaskMode classes plus a config entry

ChannelSubsampler (preprocess_params.mask.subsample) is independent of the strategy: it
REMOVES channels (they become padding, no loss) to imitate sparse caps.
"""
import torch
from abc import ABC, abstractmethod
from typing import List, Optional


class BaseMaskingStrategy(ABC):
    multiplier: int = 1
    _epoch: int = 1

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def state(self):
        """Masks are redrawn whenever this changes. Default: never after the first draw."""
        return self.multiplier

    @abstractmethod
    def generate(self, num_channels: int, num_patches: int, valid: Optional[torch.Tensor] = None,
                 coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        """-> [multiplier, C*N] bool. coords: [C, 3] channel positions, read only by modes
        that need geometry (ChannelClusterMask)."""

    def describe(self) -> str:
        return type(self).__name__


def _valid(num_channels, num_patches, valid):
    return torch.ones(num_channels * num_patches, dtype=torch.bool) if valid is None else valid.bool()


def random_token_mask(num_channels: int, num_patches: int, ratio: float, valid: torch.Tensor,
                      time_run: int = 1) -> torch.Tensor:
    """ratio of the valid tokens, as single tokens (time_run 1) or as whole runs of time_run
    consecutive patches per channel. Why runs: patches overlap by 50% (patch_stride =
    patch_len / 2), so a single masked patch with both neighbours visible has every sample in
    the input and reconstructing it is copying; a run of >= 3 hides at least one patch fully.
    The patch axis is cut into time_run-patch blocks (random offset per window) and whole
    (channel, block) units are drawn, so the inverse of a run mask is a run mask too."""
    num_masked = int(int(valid.sum()) * ratio)
    if time_run <= 1:
        idx = valid.nonzero(as_tuple=True)[0]
        mask = torch.zeros_like(valid)
        mask[idx[torch.randperm(len(idx))[:num_masked]]] = True
        return mask
    block = (torch.arange(num_patches) + int(torch.randint(time_run, (1,)))) // time_run   # [N]
    n_blocks = int(block.max()) + 1
    unit = (torch.arange(num_channels)[:, None] * n_blocks + block[None, :]).flatten()      # [C*N]
    unit_valid = torch.zeros(num_channels * n_blocks, dtype=torch.long).index_add_(0, unit, valid.long())
    order = torch.randperm(len(unit_valid))
    order = order[unit_valid[order] > 0]
    take = order[:int((unit_valid[order].cumsum(0) < num_masked).sum()) + 1] if num_masked else order[:0]
    chosen = torch.zeros(len(unit_valid), dtype=torch.bool)
    chosen[take] = True
    return chosen[unit] & valid


class RandomMaskingStrategy(BaseMaskingStrategy):
    def __init__(self, mask_ratio: float = 0.5, time_run: int = 1):
        self.mask_ratio, self.time_run = mask_ratio, max(1, int(time_run))

    def generate(self, num_channels, num_patches, valid=None, coords=None):
        return random_token_mask(num_channels, num_patches, self.mask_ratio,
                                 _valid(num_channels, num_patches, valid), self.time_run)[None]

    def describe(self):
        return f'random ratio={self.mask_ratio} time_run={self.time_run}'


class ComplementaryMaskingStrategy(BaseMaskingStrategy):
    """Fixed 0.5 random mask; every window is shown twice, with the mask and with its
    inverse (XOR with valid, so padding stays unmasked in both), so every valid token is
    reconstructed once per pair."""
    MASK_RATIO = 0.5
    multiplier = 2

    def __init__(self, time_run: int = 1):
        self.time_run = max(1, int(time_run))

    def generate(self, num_channels, num_patches, valid=None, coords=None):
        valid = _valid(num_channels, num_patches, valid)
        mask = random_token_mask(num_channels, num_patches, self.MASK_RATIO, valid, self.time_run)
        return torch.stack([mask, mask ^ valid])

    def describe(self):
        return f'complementary ratio=0.5 time_run={self.time_run}'


class RandomToComplementaryMaskingStrategy(BaseMaskingStrategy):
    """Random masking whose ratio ramps start_ratio -> 0.5 in coarse steps (one step every
    step_every masked epochs, over ramp_epochs), then complementary 0.5 pairs for good.
    Exists because masking is the one un-softened shock at the tokenizer -> masked boundary
    (spatial/temporal mixing already ramp in through zero-init LayerScale)."""
    def __init__(self, start_ratio: float = 0.1, ramp_epochs: int = 25, step_every: int = 5,
                 time_run: int = 1):
        self.start_ratio, self.target_ratio = start_ratio, ComplementaryMaskingStrategy.MASK_RATIO
        self.ramp_epochs, self.step_every = max(1, ramp_epochs), max(1, step_every)
        self._complementary = ComplementaryMaskingStrategy(time_run)
        self.time_run = max(1, int(time_run))

    def _in_ramp(self):
        return self._epoch <= self.ramp_epochs

    def ratio(self) -> float:
        if not self._in_ramp():
            return self.target_ratio
        n_steps = max(1, self.ramp_epochs // self.step_every)
        step = min((self._epoch - 1) // self.step_every, n_steps - 1)
        return self.start_ratio if n_steps <= 1 else \
            self.start_ratio + (self.target_ratio - self.start_ratio) * step / (n_steps - 1)

    @property
    def multiplier(self):
        return 1 if self._in_ramp() else 2

    def state(self):
        return (self.ratio(), self.multiplier)

    def generate(self, num_channels, num_patches, valid=None, coords=None):
        if not self._in_ramp():
            return self._complementary.generate(num_channels, num_patches, valid)
        return random_token_mask(num_channels, num_patches, self.ratio(),
                                 _valid(num_channels, num_patches, valid), self.time_run)[None]

    def describe(self):
        return (f'random_to_complementary ratio={self.ratio():.3f} x{self.multiplier} '
                f'time_run={self.time_run}')


# ---------------------------------------------------------------------------------------
# Mixture masking: one MaskMode per window
# ---------------------------------------------------------------------------------------

class MaskMode(ABC):
    """One masking pattern: generate(valid [C, N] bool, coords [C, 3], ratio) -> [C, N],
    never True outside valid. ratio is this mode's own fraction (of channels or of time)."""
    complementary = False   # True: after the ramp the window is also shown with valid ^ mask

    @abstractmethod
    def generate(self, valid: torch.Tensor, coords: Optional[torch.Tensor], ratio: float) -> torch.Tensor:
        pass

    @staticmethod
    def _n_channels(n_valid: int, ratio: float) -> int:
        """Channels to hide: ratio of the valid ones, at least 1; exactly 1 on caps of <= 4."""
        return 1 if n_valid <= 4 else max(1, int(round(ratio * n_valid)))


class RandomChannelMask(MaskMode):
    """Whole channels, chosen at random, hidden for the whole window."""
    def __init__(self, complementary: bool = True):
        self.complementary = complementary

    def generate(self, valid, coords, ratio):
        ch = valid.any(1).nonzero().flatten()
        hidden = torch.zeros(valid.shape[0], dtype=torch.bool)
        hidden[ch[torch.randperm(len(ch))[:self._n_channels(len(ch), ratio)]]] = True
        return valid & hidden[:, None]


class ChannelClusterMask(MaskMode):
    """Scalp regions hidden for the whole window: a random valid seed electrode plus every
    valid channel within a random radius (radius_cm; coords are in metres), repeated until
    the target channel count is reached. A radius rather than k-nearest keeps the physical
    size of the hole the same on a 19- and a 128-channel cap."""
    def __init__(self, radius_cm=(3.0, 7.0)):
        self.radius = (radius_cm[0] / 100.0, radius_cm[1] / 100.0)

    def generate(self, valid, coords, ratio):
        ch = valid.any(1).nonzero().flatten()
        target = self._n_channels(len(ch), ratio)
        hidden = torch.zeros(valid.shape[0], dtype=torch.bool)
        if len(ch) <= 4 or coords is None:
            hidden[ch[torch.randint(len(ch), (1,))]] = True
            return valid & hidden[:, None]
        while int(hidden[ch].sum()) < target:
            free = ch[~hidden[ch]]
            seed = free[torch.randint(len(free), (1,))]
            r = self.radius[0] + (self.radius[1] - self.radius[0]) * float(torch.rand(1))
            hidden[ch[(coords[ch] - coords[seed]).norm(dim=-1) <= r]] = True
        return valid & hidden[:, None]


class TimeBlockMask(MaskMode):
    """Blocks of run_patches consecutive patches hidden on every channel at once, until
    ratio of the window's real patches is covered (>= 3: patches overlap by 50%, a shorter
    hole is visible through its neighbours)."""
    def __init__(self, run_patches=(3, 8)):
        self.run = (int(run_patches[0]), int(run_patches[1]))

    def generate(self, valid, coords, ratio):
        real = valid.any(0).nonzero().flatten()
        lo, hi = int(real.min()), int(real.max()) + 1        # a window's real content is one span
        target = max(1, int(round(ratio * (hi - lo))))
        hidden = torch.zeros(valid.shape[1], dtype=torch.bool)
        for _ in range(100):
            if int(hidden.sum()) >= target:
                break
            L = min(int(torch.randint(self.run[0], self.run[1] + 1, (1,))), hi - lo)
            start = lo + int(torch.randint(hi - lo - L + 1, (1,)))
            hidden[start:start + L] = True
        return valid & hidden[None, :]


MASK_MODES = {'random_channel': RandomChannelMask, 'channel_cluster': ChannelClusterMask,
              'time_block': TimeBlockMask}


class MixtureMaskingStrategy(BaseMaskingStrategy):
    """Draws one MaskMode per window (shares `prob`). All modes share one ramp: over the
    first ramp_epochs masked epochs each mode's ratio goes start_ratio -> its own max_ratio
    together. After the ramp, a window whose mode is complementary is also shown with its
    inverse (the dataset doubles; other windows get a second independent draw). Masks are
    redrawn every epoch. Config: preprocess_params.mask.mixture = {start_ratio,
    ramp_epochs, modes: [{type, prob, max_ratio, ...mode kwargs}]}."""
    def __init__(self, modes: List[dict], start_ratio: float = 0.1, ramp_epochs: int = 10):
        self.modes = [MASK_MODES[m['type']](**{k: v for k, v in m.items() if k not in ('type', 'prob', 'max_ratio')})
                      for m in modes]
        self.names = [m['type'] for m in modes]
        self.probs = torch.tensor([float(m['prob']) for m in modes])
        self.max_ratio = [float(m['max_ratio']) for m in modes]
        self.start_ratio, self.ramp_epochs = start_ratio, max(1, ramp_epochs)

    def progress(self) -> float:
        return 1.0 if self.ramp_epochs <= 1 else min(1.0, (self._epoch - 1) / (self.ramp_epochs - 1))

    def ratios(self) -> List[float]:
        p = self.progress()
        return [self.start_ratio + p * (m - self.start_ratio) for m in self.max_ratio]

    @property
    def multiplier(self):
        return 2 if self._epoch > self.ramp_epochs and any(m.complementary for m in self.modes) else 1

    def state(self):
        return self._epoch   # fresh masks every epoch

    def generate(self, num_channels, num_patches, valid=None, coords=None):
        valid = _valid(num_channels, num_patches, valid).view(num_channels, num_patches)
        i = int(torch.multinomial(self.probs, 1))
        mode, ratio = self.modes[i], self.ratios()[i]
        masks = [mode.generate(valid, coords, ratio)]
        if self.multiplier == 2:
            masks.append((valid ^ masks[0]) if mode.complementary else mode.generate(valid, coords, ratio))
        return torch.stack([m.flatten() for m in masks])

    def describe(self):
        return 'mixture x{} '.format(self.multiplier) + ' '.join(
            f'{n}:{p:.2f}@{r:.2f}' for n, p, r in zip(self.names, self.probs.tolist(), self.ratios()))


class ChannelSubsampler:
    """Removes channels (not masks them) so pretraining sees sparse caps: a window whose cap
    has >= dense_min_channels real channels keeps, with probability prob(), only one montage
    drawn uniformly from `montages` -- names of configs/montages.json entries; the dataset
    resolves them to channel indices (channels the cap lacks are skipped; fewer than min_keep
    left = no subsampling). Removed channels become padding: zero signal, not valid, no loss.
    prob ramps 0 -> prob over ramp_epochs masked epochs from start_epoch."""
    def __init__(self, montages: List[str], prob: float = 0.2, start_epoch: int = 11,
                 ramp_epochs: int = 10, dense_min_channels: int = 32, min_keep: int = 3):
        self.montages = list(montages)
        self.prob_max, self.start, self.ramp = prob, start_epoch, max(1, ramp_epochs)
        self.dense_min, self.min_keep, self._epoch = dense_min_channels, min_keep, 1

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def prob(self) -> float:
        return 0.0 if self._epoch < self.start else \
            self.prob_max * min(1.0, (self._epoch - self.start + 1) / self.ramp)

    def state(self):
        return self._epoch if self.prob() > 0 else 0

    def sample(self, valid_channels: torch.Tensor, montage_idx: List[List[int]]) -> Optional[torch.Tensor]:
        """montage_idx: self.montages as canonical channel indices. -> keep [C] bool, or None."""
        if int(valid_channels.sum()) < self.dense_min or float(torch.rand(1)) >= self.prob():
            return None
        idx = [i for i in montage_idx[int(torch.randint(len(montage_idx), (1,)))] if valid_channels[i]]
        if len(idx) < self.min_keep:
            return None
        keep = torch.zeros_like(valid_channels, dtype=torch.bool)
        keep[idx] = True
        return keep

    def describe(self) -> str:
        return f'subsample prob={self.prob():.3f} montages={self.montages}'


def build_masking_strategy_from_config(pp: dict) -> BaseMaskingStrategy:
    """pp = preprocess_params.mask. time_run (default 1) applies to the token strategies."""
    name, time_run = pp.get('masking_strategy', 'random'), pp.get('time_run', 1)
    cfg = pp.get(name, {})
    if name == 'mixture':
        return MixtureMaskingStrategy(**cfg)
    if name == 'random_to_complementary':
        return RandomToComplementaryMaskingStrategy(start_ratio=cfg.get('start_ratio', 0.1),
                                                    ramp_epochs=cfg.get('ramp_epochs', 25),
                                                    step_every=cfg.get('step_every', 5), time_run=time_run)
    if name == 'complementary':
        return ComplementaryMaskingStrategy(time_run)
    return RandomMaskingStrategy(cfg.get('mask_ratio', 0.5), time_run)


def build_subsampler_from_config(pp: dict) -> Optional[ChannelSubsampler]:
    """preprocess_params.mask.subsample -> ChannelSubsampler, or None unless enabled: true."""
    cfg = dict(pp.get('subsample') or {})
    return ChannelSubsampler(**cfg) if cfg.pop('enabled', False) else None
