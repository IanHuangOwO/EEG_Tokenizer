"""Panel discovery and dispatch shared by analysis_pretrain.py and analysis_finetune.py.

A panel is a module tools/panels/panel_<name>.py exposing
  STAGES: frozenset of 'pretrain' / 'finetune' -- which entrypoint may run it
  run(ctx) -> None
Discovery is by filename only: adding a panel is adding a file. PRESETS name the panel lists the
entrypoints run by default."""
import glob
import importlib
import os
import traceback
from dataclasses import dataclass, field
from typing import Optional

PANELS_DIR = os.path.dirname(os.path.abspath(__file__))

PRESETS = {
    'pretrain': {
        'standard': ['backbone_eval', 'attention_range', 'stamp_usage', 'ridge_probe', 'stamp_templates',
                     'stamp_duplicates', 'stamp_distribution', 'snapshot', 'codebook'],
        'quick': ['backbone_eval', 'attention_range', 'stamp_usage', 'ridge_probe', 'stamp_templates',
                  'stamp_duplicates'],
    },
    'finetune': {
        'standard': ['summary', 'report', 'time_weights'],
    },
}


@dataclass
class PanelContext:
    """One per entrypoint call, passed to every selected panel.
    pretrain: config is the backbone's run config (plus the --config overlay), checkpoint its .pth,
    `model` builds lazily from the checkpoint on first use. finetune: groups {name: backbone},
    ref (a group name) and head (the finetune label, output/<backbone>/finetune/<head>/<cell>)."""
    stage: str
    config: dict
    out_dir: str
    args: object                       # argparse.Namespace: panel-specific options
    device: object = None
    checkpoint: Optional[str] = None
    groups: dict = field(default_factory=dict)
    ref: Optional[str] = None
    head: Optional[str] = None
    _model: object = None

    @property
    def model(self):
        if self._model is None:
            from tools.analysis import load_model
            self._model = load_model(self.config, self.checkpoint, self.device, mode='pretrain')
        return self._model

    @property
    def cmap(self):
        return self.config.get('training_params', {}).get('visualize_params', {}).get('cmap', 'YlOrRd')


def discover_panel_names(stage=None):
    names = sorted(os.path.basename(p)[len('panel_'):-len('.py')]
                   for p in glob.glob(os.path.join(PANELS_DIR, 'panel_*.py')))
    return [n for n in names if stage is None or stage in load_panel(n).STAGES]


def load_panel(name):
    try:
        return importlib.import_module(f'tools.panels.panel_{name}')
    except ModuleNotFoundError as e:
        if e.name != f'tools.panels.panel_{name}':
            raise
        raise ValueError(f"no panel named {name!r}; available: {discover_panel_names()}") from e


def resolve_panels(stage, preset, panels):
    """--panel names if given, else the preset's list; each checked against the stage."""
    names = panels or PRESETS[stage][preset]
    for n in names:
        if stage not in load_panel(n).STAGES:
            raise ValueError(f"panel {n!r} does not run in the {stage} stage "
                             f"(available: {discover_panel_names(stage)})")
    return names


def run_panels(names, ctx):
    """Runs each panel; a failing panel prints its traceback and the rest still run.
    Returns the names that failed (the entrypoint exits non-zero if any did)."""
    failed = []
    for n in names:
        print(f"\n=== panel {n} ===", flush=True)
        try:
            load_panel(n).run(ctx)
        except Exception:
            traceback.print_exc()
            failed.append(n)
    if failed:
        print(f"\nFAILED panels: {failed}")
    return failed
