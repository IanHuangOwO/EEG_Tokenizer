import torch
from abc import ABC, abstractmethod
from typing import List, Tuple


class BaseMaskingStrategy(ABC):
    multiplier: int = 1  # dataset-size multiplier this strategy yields (e.g. complementary pairs double it)

    @abstractmethod
    def generate_mask(self, num_channels: int, num_patches: int, mask_ratio: float) -> torch.Tensor:
        """Returns a boolean mask of shape (num_channels * num_patches,)."""
        pass

    def effective_mask_ratio(self, requested_ratio: float) -> float:
        """Ratio actually used to generate masks; override if the strategy pins its own."""
        return requested_ratio

    def resolve(self, masks: List[torch.Tensor], index: int, n: int) -> Tuple[int, torch.Tensor]:
        """Maps a dataset __getitem__ index (0..len(dataset)*multiplier-1) to (trial_idx, mask)."""
        return index, masks[index]

    def set_epoch(self, epoch: int) -> None:
        """No-op for strategies that don't vary per epoch; curriculum strategies override."""
        pass


class RandomMaskingStrategy(BaseMaskingStrategy):
    def generate_mask(self, num_channels: int, num_patches: int, mask_ratio: float) -> torch.Tensor:
        num_tokens = num_channels * num_patches
        num_masked = int(num_tokens * mask_ratio)
        indices = torch.randperm(num_tokens)
        mask = torch.zeros(num_tokens, dtype=torch.bool)
        mask[indices[:num_masked]] = True
        return mask


class ComplementaryMaskingStrategy(BaseMaskingStrategy):
    """
    Fixed mask_ratio=0.5. Dataset doubles: first half uses the mask,
    second half uses its bitwise inverse so every patch is seen as both
    masked and visible across the pair.
    """
    MASK_RATIO = 0.5
    multiplier = 2

    def generate_mask(self, num_channels: int, num_patches: int, mask_ratio: float = 0.5) -> torch.Tensor:
        return RandomMaskingStrategy().generate_mask(num_channels, num_patches, self.MASK_RATIO)

    def effective_mask_ratio(self, requested_ratio: float) -> float:
        return self.MASK_RATIO

    def resolve(self, masks: List[torch.Tensor], index: int, n: int) -> Tuple[int, torch.Tensor]:
        trial_idx = index % n
        mask = masks[trial_idx]
        if index >= n:
            mask = ~mask
        return trial_idx, mask


class RandomToComplementaryMaskingStrategy(BaseMaskingStrategy):
    """
    Curriculum strategy: ramps a RandomMaskingStrategy's ratio from
    start_ratio up to target_ratio in coarse steps (one new ratio every
    step_every epochs) over ramp_epochs total, then permanently switches to
    ComplementaryMaskingStrategy's fixed 0.5 paired masking. Caller must call
    set_epoch(epoch) before each epoch's IO.dataset.PretrainDataset.set_masking()
    so effective_mask_ratio/generate_mask/multiplier reflect the current step.

    Exists because masking is the one un-softened distribution shock at the
    Tokenizer->Masked stage boundary: spatial/temporal mixing already ramps in
    gradually via zero-init LayerScale (TSABlock.enable_temporal/enable_spatial),
    but bool_masked_pos substitutes mask_token wholesale with no ramp, and
    MeSAE's StampBank generator is now content-conditioned (docs/adr/0009's
    later revision) rather than just amplitude-conditioned — a sudden,
    never-seen-in-Tokenizer-stage input distribution is a sharper shock to a
    nonlinear generator than it was to the old linear decode chain.
    """
    def __init__(self, target_ratio: float = ComplementaryMaskingStrategy.MASK_RATIO,
                 start_ratio: float = 0.1, ramp_epochs: int = 25, step_every: int = 5):
        self.target_ratio = target_ratio
        self.start_ratio = start_ratio
        self.ramp_epochs = max(1, ramp_epochs)
        self.step_every = max(1, step_every)
        self._complementary = ComplementaryMaskingStrategy()
        self._epoch = 1

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def _in_ramp(self) -> bool:
        return self._epoch <= self.ramp_epochs

    @property
    def multiplier(self) -> int:
        return 1 if self._in_ramp() else self._complementary.multiplier

    def effective_mask_ratio(self, requested_ratio: float) -> float:
        if not self._in_ramp():
            return self._complementary.effective_mask_ratio(requested_ratio)
        n_steps = max(1, self.ramp_epochs // self.step_every)
        step = min((self._epoch - 1) // self.step_every, n_steps - 1)
        if n_steps <= 1:
            return self.start_ratio
        return self.start_ratio + (self.target_ratio - self.start_ratio) * step / (n_steps - 1)

    def generate_mask(self, num_channels: int, num_patches: int, mask_ratio: float) -> torch.Tensor:
        if not self._in_ramp():
            return self._complementary.generate_mask(num_channels, num_patches, mask_ratio)
        return RandomMaskingStrategy().generate_mask(num_channels, num_patches, mask_ratio)

    def resolve(self, masks: List[torch.Tensor], index: int, n: int) -> Tuple[int, torch.Tensor]:
        if not self._in_ramp():
            return self._complementary.resolve(masks, index, n)
        return index, masks[index]


def build_masking_strategy_from_config(strat_name, pp):
    """Constructs the configured masking strategy once, up front, from
    preprocess_params.mask (`pp`). Only 'random_to_complementary' varies per epoch
    afterward (via its own set_epoch()) — 'random'/'complementary' are constant for the
    whole run, same as before curriculum existed."""
    if strat_name == 'random_to_complementary':
        curriculum = pp.get('random_to_complementary', {})
        return RandomToComplementaryMaskingStrategy(
            target_ratio=ComplementaryMaskingStrategy.MASK_RATIO,
            start_ratio=curriculum.get('start_ratio', 0.1),
            ramp_epochs=curriculum.get('ramp_epochs', 25),
            step_every=curriculum.get('step_every', 5),
        )
    if strat_name == 'complementary':
        return ComplementaryMaskingStrategy()
    return RandomMaskingStrategy()
