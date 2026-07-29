"""
BasePlugin: the bundle a model dir (model/<Name>/plugin.py) builds at module bottom and
registers in model/factory.py's MODEL_REGISTRY. See docs/adr/0004-model-plugin-base-classes.md.
"""

from dataclasses import dataclass
from typing import Callable, Type

from model.base_trainer import BaseTrainer
from model.base_checker import BaseEpochChecker
from model.base_plotter import BasePlotter


@dataclass(frozen=True)
class BasePlugin:
    """One instance per model_type, built at the bottom of model/<Name>/plugin.py and
    registered in model/factory.py's MODEL_REGISTRY. Bundles everything a new model needs
    to plug into the shared train_pretrain.py/train_finetune.py loops."""
    build: Callable            # (bp: dict, num_channels: int) -> nn.Module, the pretrain backbone
    finetune_cls: type         # wraps a built+loaded backbone into a classifier
    trainer_cls: Type[BaseTrainer]
    checker_cls: Type[BaseEpochChecker]
    plotter_cls: Type[BasePlotter]
