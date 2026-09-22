"""
BasePlugin: the bundle a model dir (model/<Name>/plugin.py) builds at module bottom and
registers in model/factory.py's MODEL_REGISTRY. See docs/adr/0004-model-plugin-base-classes.md.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Type

from model.base_trainer import BaseTrainer
from model.base_codebook_checker import BaseCodebookChecker
from model.base_plotter import BasePlotter


@dataclass(frozen=True)
class BasePlugin:
    """One instance per model_type, built at the bottom of model/<Name>/plugin.py and
    registered in model/factory.py's MODEL_REGISTRY. Bundles everything a new model needs
    to plug into the shared train_pretrain.py/train_finetune.py loops."""
    build: Callable            # (bp: dict, num_channels: int) -> nn.Module, the pretrain backbone
    finetune_cls: Callable     # (backbone, num_channels, num_classes, **finetune params) -> classifier
    trainer_cls: Type[BaseTrainer]
    plotter_cls: Type[BasePlotter]
    # cross-dataset codebook/vocab diagnostics (analysis_pretrain.py --analysis codebook only, see
    # model/base_codebook_checker.py) — optional, None until a model implements it.
    codebook_checker_cls: Optional[Type[BaseCodebookChecker]] = None
