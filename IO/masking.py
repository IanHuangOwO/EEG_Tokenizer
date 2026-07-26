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
