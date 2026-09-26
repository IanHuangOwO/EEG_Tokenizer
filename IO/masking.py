"""
Pretrain masking (masked phase only; the tokenizer phase ignores masks).

One object, MaskingStrategy, so the training loop and the dataset never need to know which
scheme runs:

    strategy.set_epoch(e)     # e = masked-phase epoch, 1-based; the strategy owns its schedule
    strategy.state()          # hashable; the dataset redraws its masks when this changes
    strategy.multiplier       # dataset copies per epoch (2 = every window shown twice)
    strategy.generate(C, N, valid, coords) -> [multiplier, C*N] bool, one mask per copy
    strategy.subsampler       # ChannelSubsampler or None
    strategy.describe()       # one line for the log

Tokens are channel-major (c * N + n), matching IO/dataset.py's (C, N) layout. valid (bool,
same shape) marks real content: masks are never True on a zero-padded channel or on an
assembled window's zero tail, and every ratio is a ratio of the VALID tokens -- otherwise a
heavily padded dataset (8 of 64 channels) or a window's zero tail wastes most of the budget
on content already known to be zero (docs/model-analysis-checklist.md).

Strategies (preprocess_params.mask.masking_strategy) -- all MaskingStrategy: one MaskMode
per window (random_token / random_channel / channel_cluster / time_block, ...) on one shared ramp.
New mask patterns are new MaskMode classes plus a config entry. Named presets:
  random                   random_token at a fixed ratio
  complementary            random_token at 0.5, each window shown with its inverse too
  random_to_complementary  random_token ratio ramp, then complementary (the baseline's)
  mixture                  modes as configured

ChannelSubsampler (preprocess_params.mask.subsample, enabled: true) is owned by the strategy: it
REMOVES channels (they become padding, no loss) to imitate sparse caps.
"""
import torch
from abc import ABC, abstractmethod
from typing import List, Optional


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


# ---------------------------------------------------------------------------------------
# Mixture masking: one MaskMode per window
# ---------------------------------------------------------------------------------------

class MaskMode(ABC):
    """One masking pattern: generate(valid [C, N] bool, coords [C, 3], ratio) -> [C, N],
    never True outside valid. ratio is this mode's own fraction (of channels or of time)."""
    complementary = False   # True: after the ramp the window is also shown with valid ^ mask
    channel_mode = False    # True: hides whole channels -- skipped on an already-subsampled
                             # window (ChannelSubsampler already did the channel cut there;
                             # stacking another channel-removal mode on top of e.g. motor-3's
                             # 3 real channels can leave 1-2 visible)

    @abstractmethod
    def generate(self, valid: torch.Tensor, coords: Optional[torch.Tensor], ratio: float) -> torch.Tensor:
        pass

    @staticmethod
    def _n_channels(n_valid: int, ratio: float) -> int:
        """Channels to hide: ratio of the valid ones, at least 1; exactly 1 on caps of <= 4."""
        return 1 if n_valid <= 4 else max(1, int(round(ratio * n_valid)))


class RandomChannelMask(MaskMode):
    """Whole channels, chosen at random, hidden for the whole window."""
    channel_mode = True

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
    channel_mode = True

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


class RandomTokenMask(MaskMode):
    """Random (channel, patch) tokens, or whole runs of time_run patches per channel (see
    random_token_mask); ratio = fraction of valid tokens. complementary: after the ramp the
    window is also shown with the inverse, so every valid token is reconstructed once per pair."""
    def __init__(self, time_run: int = 1, complementary: bool = False):
        self.time_run, self.complementary = max(1, int(time_run)), complementary

    def generate(self, valid, coords, ratio):
        C, N = valid.shape
        return random_token_mask(C, N, ratio, valid.flatten(), self.time_run).view(C, N)


MASK_MODES = {'random_token': RandomTokenMask, 'random_channel': RandomChannelMask,
              'channel_cluster': ChannelClusterMask, 'time_block': TimeBlockMask}


class MaskingStrategy:
    """Draws one MaskMode per window (shares `prob`). All modes share one ramp: over the first
    ramp_epochs masked epochs each mode's ratio goes start_ratio -> its own max_ratio together,
    in steps of step_every epochs (1 = every epoch; ramp_epochs 0 = at max_ratio from the start).
    After the ramp, a window whose mode is complementary is also shown with its inverse (the
    dataset doubles; other windows get a second independent draw). resample_each_epoch: fresh
    masks every epoch; False keeps one draw per ratio step (the older strategies' behaviour).
    Config: preprocess_params.mask.mixture = {start_ratio, ramp_epochs, step_every,
    resample_each_epoch, modes: [{type, prob, max_ratio, ...mode kwargs}]}; the named strategies
    random / complementary / random_to_complementary are presets (PRESETS). subsample (a
    ChannelSubsampler config) is owned here too, so the training loop and the dataset handle
    one object: set_epoch / state / multiplier / generate / subsampler / describe."""
    def __init__(self, modes: List[dict], start_ratio: float = 0.1, ramp_epochs: int = 10,
                 step_every: int = 1, resample_each_epoch: bool = True, label: str = 'mixture',
                 subsample: Optional[dict] = None):
        self._epoch = 1
        self.subsampler = ChannelSubsampler(**subsample) if subsample else None
        self.modes = [MASK_MODES[m['type']](**{k: v for k, v in m.items() if k not in ('type', 'prob', 'max_ratio')})
                      for m in modes]
        self.names = [m['type'] for m in modes]
        self.probs = torch.tensor([float(m['prob']) for m in modes])
        self.max_ratio = [float(m['max_ratio']) for m in modes]
        self.start_ratio, self.ramp_epochs = start_ratio, max(0, int(ramp_epochs))
        self.step_every, self.resample, self.label = max(1, int(step_every)), resample_each_epoch, label

    def set_epoch(self, epoch: int) -> None:
        """epoch = masked-phase epoch, 1-based; drives the ramp and the subsampler's schedule."""
        self._epoch = epoch
        if self.subsampler is not None:
            self.subsampler.set_epoch(epoch)

    def progress(self) -> float:
        """Ramp position 0..1, constant within each step_every-epoch step."""
        if self.ramp_epochs == 0:
            return 1.0
        n_steps = max(1, self.ramp_epochs // self.step_every)
        if n_steps <= 1:
            return 0.0 if self._epoch <= self.ramp_epochs else 1.0
        return min((self._epoch - 1) // self.step_every, n_steps - 1) / (n_steps - 1)

    def ratios(self) -> List[float]:
        p = self.progress()
        return [self.start_ratio + p * (m - self.start_ratio) for m in self.max_ratio]

    @property
    def multiplier(self):
        return 2 if self._epoch > self.ramp_epochs and any(m.complementary for m in self.modes) else 1

    def state(self):
        """Masks (and subsampled montages) are redrawn whenever this changes."""
        own = self._epoch if self.resample else (tuple(self.ratios()), self.multiplier)
        return own, (self.subsampler.state() if self.subsampler is not None else None)

    def generate(self, num_channels, num_patches, valid=None, coords=None, subsampled=False):
        """subsampled: True when ChannelSubsampler already cut this window's channels -- a
        channel_mode mode stacked on top of that can leave almost nothing visible (e.g.
        motor-3's 3 real channels, halved again), so those modes are excluded here and the
        draw falls back to the remaining (non-channel) modes, renormalized."""
        valid = _valid(num_channels, num_patches, valid).view(num_channels, num_patches)
        probs, ratios, modes = self.probs, self.ratios(), self.modes
        if subsampled and any(m.channel_mode for m in modes):
            keep = torch.tensor([not m.channel_mode for m in modes])
            if keep.any():
                probs = probs[keep]
                modes = [m for m, k in zip(modes, keep.tolist()) if k]
                ratios = [r for r, k in zip(ratios, keep.tolist()) if k]
            else:
                return torch.zeros(self.multiplier, num_channels * num_patches, dtype=torch.bool)
        i = int(torch.multinomial(probs, 1))
        mode, ratio = modes[i], ratios[i]
        masks = [mode.generate(valid, coords, ratio)]
        if self.multiplier == 2:
            masks.append((valid ^ masks[0]) if mode.complementary else mode.generate(valid, coords, ratio))
        return torch.stack([m.flatten() for m in masks])

    def describe(self):
        return f'{self.label} x{self.multiplier} ' + ' '.join(
            f'{n}:{p:.2f}@{r:.2f}' for n, p, r in zip(self.names, self.probs.tolist(), self.ratios())) + \
            (f' | {self.subsampler.describe()}' if self.subsampler is not None else '')


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


def _token(ratio, complementary):
    return [{'type': 'random_token', 'prob': 1.0, 'max_ratio': ratio, 'complementary': complementary}]


# Named strategies = MaskingStrategy configs of a single random_token mode. cfg is
# preprocess_params.mask.<name>; time_run (preprocess_params.mask.time_run) is added to the mode.
PRESETS = {
    'random': lambda cfg: dict(modes=_token(cfg.get('mask_ratio', 0.5), False),
                               start_ratio=cfg.get('mask_ratio', 0.5), ramp_epochs=0, resample_each_epoch=False),
    'complementary': lambda cfg: dict(modes=_token(0.5, True), start_ratio=0.5, ramp_epochs=0,
                                      resample_each_epoch=False),
    'random_to_complementary': lambda cfg: dict(modes=_token(0.5, True), start_ratio=cfg.get('start_ratio', 0.1),
                                                ramp_epochs=cfg.get('ramp_epochs', 25),
                                                step_every=cfg.get('step_every', 5), resample_each_epoch=False),
}


def build_masking_strategy_from_config(pp: dict) -> MaskingStrategy:
    """pp = preprocess_params.mask: masking_strategy names a PRESETS entry or 'mixture'
    (pp['mixture'] as given); subsample runs only with enabled: true."""
    name = pp.get('masking_strategy', 'random')
    if name == 'mixture':
        cfg = dict(pp['mixture'])
    else:
        cfg = dict(PRESETS[name](pp.get(name, {})), label=name)
        cfg['modes'] = [dict(m, time_run=pp.get('time_run', 1)) for m in cfg['modes']]
    sub = dict(pp.get('subsample') or {})
    return MaskingStrategy(**cfg, subsample=sub if sub.pop('enabled', False) else None)
