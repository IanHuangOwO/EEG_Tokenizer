# Adding a new model

Protocol for wiring a third (or Nth) tokenizer model into the shared training/viz
infrastructure. Read `docs/adr/0004-model-plugin-base-classes.md` first — this doc is the
mechanical checklist that ADR's design implies.

**The rule**: a new model touches only files under `model/<Name>/` plus one line in
`model/factory.py`. If you find yourself editing `train_tokenizer.py`, `train_pretrain.py`,
`train_finetune.py`, `model/base_trainer.py`, `model/base_checker.py`, or
`model/base_plotter.py`, stop — either the new model needs a real base-class contract
change (rare, discuss first) or you're solving it the wrong way.

Pretraining runs as two sequential scripts/checkpoints, not two phases of one run: see
`CLAUDE.md`'s Architecture section. `train_tokenizer.py` builds the model and calls
`trainer.on_tokenizer_start()` (generic — enables spatial/temporal, no per-model override
needed), trains unmasked, saves a Tokenizer-stage checkpoint. `train_pretrain.py` loads
that checkpoint and calls `trainer.on_pretrain_start()` to freeze whatever the Tokenizer
stage owns, then trains masked.

## 0. Before writing code

Answer these — they shape steps 2-3 below:

- Does the model need a **Tokenizer-stage-only component to freeze** before masked
  training (VQ+decoder for MeFSQ, SAE for MeSAE)? If not, `on_pretrain_start` can be a
  no-op.
- What per-Expert/Filter/Unit **health metrics** does it need on the dashboard? List them
  now — they become `MeXXXPlotter`'s panel specs in step 3.
- What per-step **loss hyperparameters** does it need beyond `masked_mse_weight`/
  `unmasked_mse_weight` (which every model gets for free)? These become keys under
  `model_params.MeXXX.pretrain.loss` and land in `**hparams` in `compute_loss`.

## 1. Architecture: `model/MeXXX/MeXXX.py`

Two `nn.Module` classes. Match these contracts exactly — the shared checker/trainer loops
call these methods/attributes without checking `model_type`.

### `MeXXXPretrain`

```python
def forward(self, x, coords, time_idx=None, bool_masked_pos=None, valid_channels=None):
    """
    x: [B, C, N, L], coords: [B, C, 3]
    bool_masked_pos: [B, C, N] bool or None (None = unmasked, e.g. during stage 1)
    valid_channels: [B, C] bool or None (True = real, not zero-padded, channel)
    Returns a SimpleNamespace with at least:
      recon: [B, C, N, L]
      attn:  [B, N, Q, C] — each Unit's own channel-attention weights (rows sum to 1
             per Unit) — the extraction hooks in step 2 and BaseEpochChecker's attn_topo
             panel both read this
    Plus whatever extra fields get_loss/get_metrics/extract_psd need (MeFSQ carries
    v_q_routed/v_q_shared/gate_mask_routed/lb_loss; MeSAE carries sae_hidden/aux_loss).
    """

def get_loss(self, x, recon, bool_masked_pos, masked_mse_weight=1.0, unmasked_mse_weight=1.0, **extra):
    """Returns (l_total, l_masked, l_unmasked). l_masked/l_unmasked may be a plain float
    1.0/0.0 sentinel when that half genuinely doesn't apply (see MeFSQ's unmasked-stage
    case) — the training loop calls .item() defensively either way."""

def get_metrics(self, *detached_tensors):
    """Returns a flat dict merged into the epoch's logged/plotted metrics. Called once
    per epoch (train and val) with whatever forward() output tensors this model wants to
    summarize (already .detach()'d by the caller)."""

def enable_spatial(self):
    """Turns on cross-channel mixing. Called at construction-time load (viz/__init__.py
    load_model), at the start of train_tokenizer.py (BaseTrainer.on_tokenizer_start), and
    again at the start of train_pretrain.py (same hook, re-enabling after checkpoint load
    since this is a plain flag, not persisted in the state dict)."""

def enable_temporal(self):
    """Optional — only if the model has a temporal-mixing gate to unlock separately from
    spatial (MeSAE has one, MeFSQ doesn't). Checked via hasattr(model, 'enable_temporal')
    at call sites, so omit entirely if not needed."""

def freeze_<whatever_is_tokenizer_only>(self):
    """Whatever your on_pretrain_start needs to call — name it for what it freezes
    (freeze_vq_and_decoder, freeze_sae, ...). No fixed name/signature required, since only
    your own MeXXXTrainer.on_pretrain_start calls it."""
```

### `MeXXXFinetune`

```python
def __init__(self, backbone, num_channels, num_classes, hidden=128,
             freeze_backbone=False, dropout=0.1):
    """Wraps a pretrained+loaded backbone (unmodified) with a classification head.
    Store the backbone as self.backbone — BaseEpochChecker.check_finetune reads it."""

def forward(self, x, coords, time_idx=None, pad_mask=None):
    """
    x: [B, C, N, L] (already patchified — pad_mask: [B, C, N] bool, True=valid)
    Returns (logits [B, num_classes], attn_h [B, C, Q], attn_n [B, C, Q, N], attn_c [B, C])
    attn_h is the classifier head's own per-channel attention over Units — this is what
    BaseEpochChecker.check_finetune's attn_topo panel plots (transposed), separate from
    the backbone's own attn.
    """
```

## 2. Plugin: `model/MeXXX/plugin.py`

One file, four things: a `build_model` function, and three classes subclassing the
bases in `model/base_trainer.py` / `model/base_checker.py` / `model/base_plotter.py`.
MeFSQ's and MeSAE's `plugin.py` are the reference examples — copy the shape, not
necessarily the content.

```python
"""MeXXX's implementation of the shared model-plugin contract."""

import torch

from model.MeXXX.MeXXX import MeXXXPretrain, MeXXXFinetune
from model.base_trainer import BaseTrainer
from model.base_checker import BaseEpochChecker
from model.base_plotter import BasePlotter
from model.base_plugin import BasePlugin
from viz.extract import extract_head_psd, extract_head_spectra  # or write MeXXX-specific ones,
                                                                  # returning PsdResult/SpectraResult


def build_model(bp, num_channels):
    """bp: config['model_params']['MeXXX']['pretrain']. Reads bp.get(...) with defaults,
    constructs and returns MeXXXPretrain(...)."""
    ...
    return MeXXXPretrain(..., num_channels=num_channels)


@torch.no_grad()
def _run_reconstruction(model, dataset, trial_idx, device):
    """One trial -> dict(raw, recon, coords, T, N, L, fs). See MeFSQ/plugin.py or
    MeSAE/plugin.py for the full pattern (unsqueeze batch dim, forward unmasked, reshape
    back to [C, T])."""
    ...


class MeXXXTrainer(BaseTrainer):
    def compute_loss(self, model, x, out, mp, masked_mse_weight, unmasked_mse_weight, warmup, **hparams):
        return model.get_loss(x, out.recon, mp, masked_mse_weight=masked_mse_weight,
                               unmasked_mse_weight=unmasked_mse_weight, **hparams)

    def epoch_metrics(self, model, out):
        return model.get_metrics(...)

    def on_pretrain_start(self, model, logger=None):
        # on_tokenizer_start (enable_spatial/enable_temporal) is generic — BaseTrainer
        # already re-runs it before this is called. Only freezing goes here.
        model.freeze_<whatever>()
        if logger:
            logger.info("  [Pretrain] ...")


class MeXXXChecker(BaseEpochChecker):
    unit_label = 'Unit'  # or whatever this model calls its per-patch quantity

    def extract_psd(self, model, x_in, c_in, t_in, vc_in) -> PsdResult:
        """Returns viz/extract.py's PsdResult(psd_ch_x, norms, affinity, importance) —
        see extract_head_psd/extract_filter_psd for the shape contract. Reuse one of
        those if your forward pass matches (per-Unit decode -> per-channel activation
        norm), or write a MeXXX-specific one in viz/extract.py returning the same
        PsdResult dataclass."""

    def extract_spectra(self, model, x_in, c_in, t_in, vc_in, fs, freq_resolution) -> SpectraResult:
        """Returns viz/extract.py's SpectraResult(psd [Q, C, F], freqs [F], importance [Q])."""

    def run_reconstruction(self, model, dataset, trial_idx, device):
        return _run_reconstruction(model, dataset, trial_idx, device)


class MeXXXPlotter(BasePlotter):
    def plot_pretrain(self, filename='training_dashboard.png'):
        panels = [
            dict(title='Total Loss', ylabel='Loss', series=[dict(key='loss', color='b')]),
            # ... one dict per panel, from your step-0 metrics list. Use
            # self.pool_pair_series(...) / self.indexed_series(...) for
            # routed/shared-style or unbounded-family metrics.
        ]
        self.render(panels, filename, suptitle='Training Dashboard')

    def plot_finetune(self, filename='training_dashboard.png', freeze_backbone=False):
        panels = [
            dict(title='Total Loss', ylabel='Loss', series=[dict(key='loss', color='b')]),
            dict(title='Accuracy', ylabel='Acc', series=[dict(key='acc', color='crimson')]),
            dict(title='F1 (macro)', ylabel='F1', series=[dict(key='f1', color='steelblue')]),
            # + any of your pretrain-side health panels that still make sense post-freeze
        ]
        self.render(panels, filename, suptitle='Training Dashboard')


PLUGIN = BasePlugin(
    build=build_model,
    finetune_cls=MeXXXFinetune,
    trainer_cls=MeXXXTrainer,
    checker_cls=MeXXXChecker,
    plotter_cls=MeXXXPlotter,
)
```

## 3. Register: `model/factory.py`

Two lines:

```python
from model.MeXXX.plugin import PLUGIN as MEXXX_PLUGIN
# ...
MODEL_REGISTRY = {
    'MeFSQ': MEFSQ_PLUGIN,
    'MeSAE': MESAE_PLUGIN,
    'MeXXX': MEXXX_PLUGIN,
}
```

## 4. Config

Add `model_params.MeXXX.pretrain` (architecture hyperparams + a `loss` sub-object with
`masked_mse_weight`/`unmasked_mse_weight` and any extra hyperparams `compute_loss` reads
via `**hparams`) and
`model_params.MeXXX.finetune` (`hidden`, `freeze_backbone`, `dropout`). Set
`training_params.tokenizer.model_type` / `training_params.pretrain.model_type` /
`training_params.finetune.model_type` to `"MeXXX"` to select it for a run — architecture
always comes from `model_params.MeXXX.pretrain` regardless of which stage is building it.

## 5. Verify

No test suite exists for this repo — verify by hand:

```bash
# 1. Model builds from config
python -c "
import json
from model.factory import build_pretrain_from_config
cfg = json.load(open('config/config.json'))
cfg['training_params']['pretrain']['model_type'] = 'MeXXX'
m = build_pretrain_from_config(cfg)
print(type(m).__name__, sum(p.numel() for p in m.parameters()))
"

# 2. Plotter renders from synthetic history (catches panel-spec typos without a real run)
python -c "
import tempfile
from model.MeXXX.plugin import MeXXXPlotter
p = MeXXXPlotter(output_dir=tempfile.mkdtemp())
for i in range(3):
    p.update(dict(loss=1.0-i*0.1, ...), dict(loss=1.1-i*0.1, ...))
p.plot_pretrain()
p.plot_finetune(freeze_backbone=True)
"

# 3. A short real Tokenizer run exercises Trainer.compute_loss, epoch_metrics,
#    on_tokenizer_start, and Checker.check_pretrain (via train_tokenizer.py's periodic
#    viz call) end to end.
python train_tokenizer.py --config <a config with model_type=MeXXX>

# 4. A short real Pretrain run (pointed at the checkpoint from step 3, via
#    training_params.pretrain.tokenizer_checkpoint) exercises on_pretrain_start and the
#    masked training path.
python train_pretrain.py --config <a config with model_type=MeXXX>
```
