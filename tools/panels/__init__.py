"""Panel discovery/dispatch shared by analysis_pretrain.py and analysis_finetune.py.

A panel is a module at tools/panels/panel_<name>.py exposing:
  STAGES: frozenset[str]     -- subset of {'pretrain', 'finetune'}: which analysis_*.py
                                 script(s) may run it
  NEEDS_CHECKPOINT: bool     -- whether the CLI must resolve a model before calling it
  NEEDS_DATASET: bool        -- whether the CLI must resolve a dataset before calling it
  def run(ctx) -> None       -- does the work; panels are terminal actions, not
                                 composable functions

Discovery is filename convention only -- no registry, no decorator, no base class:
adding a panel is adding a file, removing one is deleting a file."""
import glob
import importlib
import os
from dataclasses import dataclass
from typing import Optional

PANELS_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass
class PanelContext:
    """Built once per analysis_*.py CLI invocation, passed to every selected panel's
    run(). A panel with NEEDS_CHECKPOINT=NEEDS_DATASET=False simply never reads
    model/dataset/checkpoint."""
    config: dict
    output_dir: str
    device: object            # torch.device -- kept untyped here so this module doesn't
                               # need to import torch just to define the dataclass
    args: object               # argparse.Namespace, for panel-specific CLI flags
    model: Optional[object] = None
    dataset: Optional[object] = None
    checkpoint: Optional[str] = None


def discover_panel_names():
    """Sorted panel names (each tools/panels/panel_<name>.py filename, minus the
    'panel_' prefix and '.py' suffix)."""
    paths = glob.glob(os.path.join(PANELS_DIR, 'panel_*.py'))
    return sorted(os.path.basename(p)[len('panel_'):-len('.py')] for p in paths)


def load_panel(name):
    """Import tools.panels.panel_<name>, return the module."""
    return importlib.import_module(f'tools.panels.panel_{name}')


def run_panels(names, stage, ctx):
    """Loads and runs each named panel in order, checked against `stage`
    ('pretrain'/'finetune'). Raises ValueError naming the panel if it doesn't declare
    support for this stage -- a clear error beats a silent no-op."""
    for name in names:
        mod = load_panel(name)
        if stage not in mod.STAGES:
            raise ValueError(f"panel {name!r} does not support stage {stage!r} "
                              f"(STAGES={sorted(mod.STAGES)})")
        mod.run(ctx)


def any_needs(names, attr):
    """True if any named panel declares `attr` (NEEDS_CHECKPOINT/NEEDS_DATASET) True."""
    return any(getattr(load_panel(name), attr) for name in names)
