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
    bundle: Optional[object] = None   # a tools.analysis.snapshot.SnapshotBundle, for
                                       # panels that render one already-prepared trial
                                       # (kept untyped for the same reason as model/dataset
                                       # above -- this package stays import-light)
    cmap: str = 'YlOrRd'       # matplotlib colormap, shared by every panel this ctx runs


def discover_panel_names():
    """Sorted panel names (each tools/panels/panel_<name>.py filename, minus the
    'panel_' prefix and '.py' suffix)."""
    paths = glob.glob(os.path.join(PANELS_DIR, 'panel_*.py'))
    return sorted(os.path.basename(p)[len('panel_'):-len('.py')] for p in paths)


def load_panel(name):
    """Import tools.panels.panel_<name>, return the module. Raises ValueError (not a raw
    ModuleNotFoundError) naming the available panels if `name` doesn't exist."""
    try:
        return importlib.import_module(f'tools.panels.panel_{name}')
    except ModuleNotFoundError as e:
        raise ValueError(f"no panel named {name!r}; available panels: "
                          f"{discover_panel_names()}") from e


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


def build_panel_context(args, names, stage, resolve_base_path):
    """Shared context-assembly for analysis_pretrain.py's and analysis_finetune.py's
    --panel branches: validates each panel supports `stage`, resolves a checkpoint+model
    only if any selected panel needs one, and returns a PanelContext ready for
    run_panels(names, stage, ctx).

    resolve_base_path(args, overlay, checkpoint) -> str: each script's own way of finding
    the base run config when a NEEDS_CHECKPOINT panel is selected (analysis_pretrain.py
    auto-derives it from the checkpoint's directory; analysis_finetune.py requires
    --base-config explicitly, since finetune runs write a timestamped config file with no
    fixed name to guess)."""
    import json
    import torch

    from tools.analysis import _deep_merge, load_model, resolve_output_dir

    for name in names:
        if stage not in load_panel(name).STAGES:
            raise ValueError(f"panel {name!r} does not support the {stage!r} stage")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if any_needs(names, 'NEEDS_CHECKPOINT'):
        if not args.config:
            raise ValueError('--config is required unless every selected panel has '
                              'NEEDS_CHECKPOINT=False')
        with open(args.config, 'r') as f:
            overlay = json.load(f)
        checkpoint = args.checkpoint or overlay.get('checkpoint', '')
        base_path = resolve_base_path(args, overlay, checkpoint)
        with open(base_path, 'r') as f:
            base = json.load(f)
        cfg = _deep_merge(base, overlay)
        for m, dsp in overlay.get('dataset_params', {}).items():
            cfg['dataset_params'][m] = dsp
        mdl = load_model(cfg, checkpoint, device, mode=stage)
        out_dir = resolve_output_dir(cfg, 'analysis', mode=stage)
    else:
        with open('config/config.json', 'r') as f:
            cfg = json.load(f)
        checkpoint, mdl = None, None
        out_dir = 'output/tools-profile'

    if any_needs(names, 'NEEDS_DATASET'):
        raise NotImplementedError(
            "panel(s) " + repr([n for n in names if load_panel(n).NEEDS_DATASET]) +
            " need a dataset/bundle, but --panel never builds one -- only the "
            "legacy --analysis path does (it builds a SnapshotBundle via "
            "build_pretrain_bundle/build_finetune_bundle and sets ctx.bundle before "
            "calling run_panels). Use --analysis for these panels until --panel "
            "gains bundle-building support.")

    return PanelContext(config=cfg, output_dir=out_dir, device=device, args=args,
                         model=mdl, dataset=None, checkpoint=checkpoint)
