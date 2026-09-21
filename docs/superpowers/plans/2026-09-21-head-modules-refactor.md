# Finetune head modules refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `MeSAEFeatureHead`'s feature math into the swappable pieces decided in ADR 0016 (spatial mix, time pooling, phase-advance branch, evoked branch) without changing any behaviour, parameter name, buffer, RNG order or checkpoint.

**Architecture:** The pieces are small `nn.Module`s and functions whose parameter names equal the current ones (`p`, `q`). They end up at the end of `model/MeSAE/MeSAE_modules.py`, under a section header that separates them from the pretrain modules above (user decision). `MeSAEFeatureHead` (in `MeSAE.py`) keeps its constructor, attributes and state-dict keys and delegates to the pieces. Task 1 writes the pieces in a temporary new file `model/MeSAE/head_modules.py` (additive, safe while experiments run, since every queued run imports `MeSAE_modules.py`); Task 2 moves them into `MeSAE_modules.py` and rewires the head; Task 3 removes the code made obsolete by ADR 0016. Tasks 2 and 3 are gated on the Phase 2 queue finishing.

**Tech Stack:** PyTorch (`eeg_fm` env), existing `model/factory.py`, `viz/__init__.py`.

**Spec:** `docs/adr/0016-finetune-head-modules.md` (modules, parameters, hard constraint on checkpoint compatibility). Current code: `model/MeSAE/MeSAE.py` class `MeSAEFeatureHead` (read it fully first).

## Global Constraints

- **Behaviour-preserving.** No config, factory, viz or training-script change; no change to results for any existing config.
- **State dict compatibility (hard):** keys stay `head.spatial.weight`, `head.time.p`, `head.time.q`, `head.evoked.p`, `head.evoked.q`, `head.z_proj.*`, `head.cls.{0,2}.*`, buffers `keep`, `E_D`, `E_H`, and `backbone.*`. `viz.load_model` reads `m.keep`, `m.head.time.p`, `m.head.time.q`, `m.head.evoked.q`, `m.head.cls[...]`.
- **RNG order must not change.** Parameters are created in the current order: `head['spatial']` (nn.Linear), `time` p then q, `evoked` p then q, `z_proj`, `cls`. A model built under the same seed must have identical initial weights before and after the refactor.
- **Python env:** `/home/mamechin/anaconda3/envs/eeg_fm/bin/python` for everything (never `base`). CPU only for checks (`CUDA_VISIBLE_DEVICES=''`).
- **Do not disturb the running Phase 2 queue.** It starts a fresh Python process per run that imports `model/MeSAE/MeSAE.py`. Task 1 must not edit any existing file. Task 2 must not start until the file `statusp2.txt` in `/tmp/claude-1000/-media-mamechin-PortableSSD-iansaididontcare-CNElab-cnelab-model-trainer-EEG-Tokenizer/1bdcaf1d-d097-4ed2-b4e6-2e97ca3102c7/scratchpad/loso/` contains a line `ALLDONE`.
- **Line endings:** `model/MeSAE/MeSAE.py` is CRLF (1019 lines, every line has CR). Never rewrite the whole file; edit in place, and `git diff --stat` must show only the intended lines. New files are LF.
- No test suite exists (CLAUDE.md): validation is runnable self-checks and the equivalence script below, not pytest files.
- Every commit ends with:
  `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_011ZRo7FMWpgbtn1N34keF18`
  Do not push (the controller pushes). Never commit anything under `output/` or `.superpowers/`.

---

## Task 1: `head_modules.py` (additive, no existing file changes)

**Files:**
- Create: `model/MeSAE/head_modules.py`

**Interfaces:**
- Produces (used by Task 2):
  - `spatial_mix(spatial, t, dim)`: `spatial` is `nn.Linear(C, K, bias=False)` or `None` (concat: identity); mixes the channel axis `dim` of `t`, returns the same layout with `K` on that axis, float32.
  - `FlatTimePool()`: `forward(power[B, N', K, S]) -> [B, K, S]` (mean over patches).
  - `LearnedTimePool(rank, num_stamps, num_patches)`: parameters `p [R, S]`, `q [R, N']`, both `randn * 0.02`, created `p` then `q`; `forward(power) -> [B, K, S]` softmax-over-patches weighted sum; method `weights() -> [S, N']`.
  - `EvokedBranch(rank, num_stamps, num_patches)`: same parameters/init as above; `forward(a, b) -> [B, K, 2*S]` = `cat([ev_re, ev_im], -1)`.
  - `phase_advance(a, b) -> [B, K, 2*S]` = `cat([z_re, z_im], -1)`.

- [ ] **Step 1: Write the file**

```python
"""Swappable pieces of the finetune head (ADR 0016). Parameter names (p, q) and their init
match the pre-refactor MeSAEFeatureHead so saved checkpoints load unchanged.

Shapes: a, b are the spatially mixed code amplitudes [B, N', K, S] (B trials, N' patches,
K spatial filters, S alive stamps); power = a^2 + b^2."""
import torch
import torch.nn as nn


def spatial_mix(spatial, t, dim):
    """Signed spatial filter (nn.Linear(C, K, bias=False)) over channel axis `dim`, or
    identity when spatial is None (channel concat)."""
    if spatial is None:
        return t
    return torch.movedim(spatial(torch.movedim(t, dim, -1).float()), -1, dim)


class FlatTimePool(nn.Module):
    """Uniform weights over patches (ADR 0014 C0)."""
    def forward(self, power):                                    # [B, N', K, S] -> [B, K, S]
        return power.mean(1)


class LearnedTimePool(nn.Module):
    """Low-rank softmax time weights w[s, n] = softmax_n(sum_r p[r, s] q[r, n]) (ADR 0014 C1).
    Small init => starts equal to the flat mean."""
    def __init__(self, rank, num_stamps, num_patches):
        super().__init__()
        self.p = nn.Parameter(torch.randn(rank, num_stamps) * 0.02)
        self.q = nn.Parameter(torch.randn(rank, num_patches) * 0.02)

    def weights(self):                                           # [S, N']
        return torch.softmax(torch.einsum('rs,rn->sn', self.p, self.q), dim=-1)

    def forward(self, power):                                    # [B, N', K, S] -> [B, K, S]
        return torch.einsum('sn,bnks->bks', self.weights(), power)


class EvokedBranch(nn.Module):
    """Signed low-rank time filter T[s, n] = 1/N' + sum_r p[r, s] q[r, n] applied LINEARLY to
    a and b (phase-locked content survives a linear functional, not power) (ADR 0014 C4)."""
    def __init__(self, rank, num_stamps, num_patches):
        super().__init__()
        self.p = nn.Parameter(torch.randn(rank, num_stamps) * 0.02)
        self.q = nn.Parameter(torch.randn(rank, num_patches) * 0.02)

    def forward(self, a, b):                                     # -> [B, K, 2*S]
        T = 1.0 / a.shape[1] + torch.einsum('rs,rn->sn', self.p, self.q)
        return torch.cat([torch.einsum('sn,bnks->bks', T, a),
                          torch.einsum('sn,bnks->bks', T, b)], dim=-1)


def phase_advance(a, b):
    """z[k, s] = sum_n u[n+1] conj(u[n]), u = a + i b, as real/imag parts (no complex dtype)
    (ADR 0014 C3). Returns cat([z_re, z_im], -1) [B, K, 2*S]."""
    a_next, a_prev, b_next, b_prev = a[:, 1:], a[:, :-1], b[:, 1:], b[:, :-1]
    z_re = (a_next * a_prev + b_next * b_prev).sum(1)
    z_im = (b_next * a_prev - a_next * b_prev).sum(1)
    return torch.cat([z_re, z_im], dim=-1)


if __name__ == '__main__':
    # self-check against independent formulas (complex dtype, explicit softmax)
    torch.manual_seed(0)
    B, N, K, S, R = 3, 39, 8, 25, 2
    a, b = torch.randn(B, N, K, S), torch.randn(B, N, K, S)
    u = torch.complex(a, b)
    z = (u[:, 1:] * u[:, :-1].conj()).sum(1)
    pa = phase_advance(a, b)
    assert torch.allclose(pa[..., :S], z.real, atol=1e-5) and torch.allclose(pa[..., S:], z.imag, atol=1e-5)
    ltp = LearnedTimePool(R, S, N)
    nn.init.normal_(ltp.p); nn.init.normal_(ltp.q)
    w = ltp.weights()
    assert torch.allclose(w.sum(-1), torch.ones(S), atol=1e-6)
    power = a.pow(2) + b.pow(2)
    ref = torch.einsum('sn,bnks->bks', w, power)
    assert torch.allclose(ltp(power), ref, atol=1e-6)
    nn.init.zeros_(ltp.p)                                        # zero logits => uniform => flat mean
    assert torch.allclose(ltp(power), FlatTimePool()(power), atol=1e-5)
    ev = EvokedBranch(R, S, N)
    nn.init.zeros_(ev.p)                                         # T = 1/N' => trial mean of a, b
    out = ev(a, b)
    assert torch.allclose(out[..., :S], a.mean(1), atol=1e-5) and torch.allclose(out[..., S:], b.mean(1), atol=1e-5)
    lin = nn.Linear(64, K, bias=False)
    t = torch.randn(B, N, 64, S)
    assert spatial_mix(lin, t, 2).shape == (B, N, K, S) and spatial_mix(None, t, 2) is t
    assert [n for n, _ in ltp.named_parameters()] == ['p', 'q'] and [n for n, _ in ev.named_parameters()] == ['p', 'q']
    print('head_modules self-check OK')
```

- [ ] **Step 2: Run the self-check**

Run: `CUDA_VISIBLE_DEVICES='' PYTHONPATH=. /home/mamechin/anaconda3/envs/eeg_fm/bin/python model/MeSAE/head_modules.py`
Expected: `head_modules self-check OK`. If the import of `model.MeSAE` package is needed, run it as `python -m model.MeSAE.head_modules` instead.

- [ ] **Step 3: Confirm nothing else changed**

Run: `git status --short`
Expected: only `?? model/MeSAE/head_modules.py`.

- [ ] **Step 4: Commit**

```bash
git add model/MeSAE/head_modules.py
git commit -m "feat: swappable finetune head pieces (spatial mix, time pool, evoked, phase advance)"
```
(with the two attribution lines)

---

## Task 2: Move the pieces into `MeSAE_modules.py` and wire `MeSAEFeatureHead` to them (gated on `ALLDONE`)

**Files:**
- Modify: `model/MeSAE/MeSAE_modules.py` (LF, 1404 lines; append the pieces at the end)
- Delete: `model/MeSAE/head_modules.py`
- Modify: `model/MeSAE/MeSAE.py` (CRLF; class `MeSAEFeatureHead` only)
- Create (not committed): `.superpowers/sdd/2026-09-21-head-modules-refactor/head_equiv.py`

**Interfaces:**
- Consumes: everything Task 1 produces.
- Produces: unchanged public behaviour: constructor arguments, attributes (`keep`, `window`, `time_rank`, `include_advance`, `evoked_rank`, `raw_erp`, `input`), `head` ModuleDict keys, `_mix`, `forward`.

- [ ] **Step 0: Gate.** Check `statusp2.txt` (path in Global Constraints) contains `ALLDONE`. If not, stop and report; do not edit anything.

- [ ] **Step 1: Save the pre-refactor class for comparison**

Run: `git show HEAD:model/MeSAE/MeSAE.py > model/MeSAE/_old_mesae_tmp.py` (CRLF preserved; this temp file is never committed and is deleted in Step 6). Confirm `python -c "import model.MeSAE._old_mesae_tmp"` imports (it uses the same relative imports as `MeSAE.py`).

- [ ] **Step 2: Write the equivalence script** at `.superpowers/sdd/2026-09-21-head-modules-refactor/head_equiv.py`. It must:
  1. Build the real backbone once: `build_pretrain_from_config` from `output/pretrain/mesae_v10_small/artifacts/config.json`, load `output/pretrain/mesae_v10_small/checkpoint/last.pth` (strict=False), eval, CPU.
  2. For each config in this list build the old class (`model.MeSAE._old_mesae_tmp.MeSAEFeatureHead`) and the new class (`model.MeSAE.MeSAE.MeSAEFeatureHead`) with `torch.manual_seed(0)` before each construction, `num_channels=64`, `num_classes=4`, `num_patches=39`, `dropout=0.5`:
     - `raw`, mi, `concat`, `trial`; `raw`, mi, `spatial:8`, `trial`; `raw`, erp, `spatial:8`, `trial`
     - `recon`, mi, `spatial:8`, `trial`; `stamp_bandpow`, mi, `spatial:8`, `trial`; `z_chan`, mi, `spatial:8`, `trial`
     - `stamp_induced` with each of: `trial`; `learned:2`; `window:1.0-4.0`; `learned:2` + `include_advance`; `learned:2` + `evoked_rank=2`; `learned:2` + both
  3. Assert **before any forward** that `sorted(old.state_dict().keys()) == sorted(new.state_dict().keys())` and every tensor is equal (`torch.equal`): this proves identical parameter names, buffers and RNG order.
  4. Put both in `eval()`; on a fixed random input `x [2, 64, 39, 50]`, `coords [2, 64, 3]` (any fixed values), `valid_channels` with 22 True and 42 False, `time_idx=None`, assert `torch.allclose(old(...)[0], new(...)[0], atol=1e-6)`; also assert the backward gradient of `logits.sum()` matches for every trainable parameter (`atol=1e-6`).
  5. Real checkpoint: load `output/experiment_c/mesae_finetune_c1_learned2/finetune/fold1_subj_8/best_finetune.pth` into both (config from `output/experiment_c/mesae_finetune_c1_learned2/artifacts/config.json` via `viz.load_config` / `viz.load_model` style build; strict), and assert equal logits on the same input.
  Print one line per configuration `OK <config>`.

- [ ] **Step 3: Run it against the unmodified class first** (both sides are still the same code): `CUDA_VISIBLE_DEVICES='' PYTHONPATH=. /home/mamechin/anaconda3/envs/eeg_fm/bin/python .superpowers/sdd/2026-09-21-head-modules-refactor/head_equiv.py`
Expected: all `OK`. This proves the harness itself is valid before the refactor.

- [ ] **Step 4: Refactor `MeSAEFeatureHead`** in `model/MeSAE/MeSAE.py` (CRLF, in place, small edits only):
  - **Move the pieces (do this first).** Append the contents of `head_modules.py` (everything except its `__main__` self-check and module docstring, whose imports already exist in `MeSAE_modules.py`) to the end of `model/MeSAE/MeSAE_modules.py`, after `StampBank`/`PerChannelHeadAttn`, under this section header in the file's existing style:
    ```
    # ==========================================
    # FINETUNE HEAD MODULES (ADR 0016)
    # Everything above this line is the pretrain side.
    # ==========================================
    ```
    Keep the self-check as a short runnable block inside a `_selfcheck_head_modules()` function at the bottom of that section (not run on import), run it once via `python -c "from model.MeSAE.MeSAE_modules import _selfcheck_head_modules as f; f()"` and expect `head_modules self-check OK`. Then `git rm model/MeSAE/head_modules.py`.
  - import: `from .MeSAE_modules import spatial_mix, FlatTimePool, LearnedTimePool, EvokedBranch, phase_advance` (extend the existing `from .MeSAE_modules import ...` line).
  - In `__init__`: replace the `head['time'] = nn.ParameterDict({...})` block with `head['time'] = LearnedTimePool(R, S, num_patches)` and the `head['evoked'] = nn.ParameterDict({...})` block with `head['evoked'] = EvokedBranch(self.evoked_rank, len(keep), num_patches)`. Keep the surrounding comments that explain the small init (move the sentences into the two classes' docstrings only if they are not already there). The construction order must not change.
  - Add `self.flat_pool = FlatTimePool()` (parameter-free; adds no state-dict keys).
  - `_mix`: `return spatial_mix(self.head['spatial'] if 'spatial' in self.head else None, t, dim)`.
  - In `forward`, inside the `stamp_induced` branch, replace the body after `power = a.pow(2) + b.pow(2)` with:
    ```python
    pool = self.head['time'] if self.time_rank is not None else self.flat_pool
    feat = torch.log(pool(power) + 1e-12)                        # [B, K, S]
    if self.include_advance:
        feat = torch.cat([feat, phase_advance(a, b)], dim=-1)    # ADR 0014 C3
    if self.evoked_rank:
        feat = torch.cat([feat, self.head['evoked'](a, b)], dim=-1)   # ADR 0014 C4
    ```
    Keep one short comment per branch pointing at ADR 0014 C1/C3/C4 and the reason `feat` is fed raw (the shared BatchNorm normalises scale); the long derivations move to the module docstrings.
  - Update the class docstring's `include_advance` / `evoked_rank` paragraphs only where they describe implementation location (add "implemented in head_modules.py").
  Preserve CRLF: edit with a script that reads/writes bytes and converts only the replaced region, then confirm `grep -c $'\r' model/MeSAE/MeSAE.py` equals `wc -l`, and `git diff --stat` shows a small diff (order of tens of lines).

- [ ] **Step 5: Re-run the equivalence script**

Run the Step 3 command. Expected: every configuration prints `OK`, including the real-checkpoint check. Any mismatch is a bug in the refactor, not in the test: fix the refactor. Then run this smoke check (no training run is needed): `CUDA_VISIBLE_DEVICES='' PYTHONPATH=. /home/mamechin/anaconda3/envs/eeg_fm/bin/python -c "import json; from viz import load_config, load_model; import torch; c=load_config('output/experiment_c/mesae_finetune_c1_learned2/artifacts/config.json'); m=load_model(c,'output/experiment_c/mesae_finetune_c1_learned2/finetune/fold1_subj_8/best_finetune.pth',torch.device('cpu'),mode='finetune'); print(type(m).__name__, m.head.time.p.shape, m.keep.shape)"` (expects `MeSAEFeatureHead torch.Size([2, 25]) torch.Size([25])`).

- [ ] **Step 6: Clean up and commit**

Delete `model/MeSAE/_old_mesae_tmp.py`. `git status --short` must show only `M model/MeSAE/MeSAE.py`, `M model/MeSAE/MeSAE_modules.py` and `D model/MeSAE/head_modules.py`. Commit:
```bash
git add model/MeSAE/MeSAE.py model/MeSAE/MeSAE_modules.py
git rm --cached -q model/MeSAE/head_modules.py 2>/dev/null || true
git commit -m "refactor: MeSAEFeatureHead delegates to head_modules pieces (behaviour and checkpoints unchanged)"
```
(with the two attribution lines)

- [ ] **Step 7: Note in ADR 0016** (LF): under Consequences add one sentence: "Implemented at the end of `model/MeSAE/MeSAE_modules.py` (finetune section); equivalence with the previous class verified on 12 configurations plus a saved C1 checkpoint." and change nothing else. Commit separately: `docs: ADR 0016 notes the head modules refactor landed`.

---

## Task 3: Remove code made obsolete by ADR 0016 (gated: after Task 2 is committed)

Scope decided by the user: the original `head_z` head path and the `MeSAEFeatureHead` options ADR 0016 drops; the old Experiment A probes are deleted; pretrain-side methods stay. The last commit that still contains all of it is tagged `pre-head-cleanup` (already pushed); reproducing Experiments A/B or loading their checkpoints needs that tag.

**Files:**
- Modify: `model/MeSAE/MeSAE.py` (CRLF): delete class `MeSAEFeatureHead` options `input in ('recon', 'z_chan')`, the `z_proj` parameter and branch, and channel-concat pooling (`pool_channel` must be `spatial:K`, else `ValueError`; also delete the concat-only "zero padded channels after the log" block in `forward`); make the constructor defaults `input='stamp_induced'`, `pool_channel='spatial:8'`; delete class `MeSAEFinetune`; drop `PerChannelHeadAttn` from the import; make `build_finetune` build only `MeSAEFeatureHead` (default `input='stamp_induced'`, remove the `head_z` branch, remove `z_proj` from `allowed`); fix docstrings that mention the removed pieces. Keep `raw` (band power and erp), `stamp_bandpow`, `stamp_induced`, `learned:R`, `window:lo-hi`, `include_advance`, `evoked_rank`, `task`. Keep every `MeSAEPretrain` method (`encode_post_stamp_expert`, `encode_used_stamps`, ...): the pretrain viz uses them.
- Modify: `model/MeSAE/MeSAE_modules.py` (LF): delete class `PerChannelHeadAttn` (the section header from Task 2 stays directly after `StampBank`); update the comment at about line 1001 that mentions `MeSAEFinetune.encode_post_stamp_expert`.
- Modify: `model/MeSAE/plugin.py` (CRLF check first): drop `MeSAEFinetune` from the import; delete `render_finetune_attn` and the code that only exists for it, and any `check_finetune` path in `model/base_checker.py` that depends on `MeSAEFinetune`'s attention output (`MeSAEFeatureHead.forward` already returns `(logits, None, None)`, so those branches are dead); keep `check_finetune` working for `MeSAEFeatureHead` (read `check_model.py:130`, `train_finetune.py:562`, `base_checker.py` first; if removing a branch would change what `train_finetune.py` calls, stop and report).
- Delete: `probes/probe_v10.py`, `probes/stamp_relevance.py`, and their rows in `probes/README.md`.
- Modify: `CLAUDE.md` (Architecture bullets that name `PerChannelHeadAttn` and `MeSAEFinetune`, lines about 88 to 90), `CONTEXT.md` line 29 (drop the `PerChannelHeadAttn` example), `docs/adr/0016-finetune-head-modules.md` (Consequences: one sentence "Obsolete paths removed; the last commit with them is tagged `pre-head-cleanup`").
- Not touched: ADR 0012/0014 text and `docs/adr/0014_attempts.csv` (historical record), `output/`.

- [ ] **Step 1: Confirm the surface.** Run `git grep -n -E "MeSAEFinetune|PerChannelHeadAttn|head_z|z_chan|z_proj|use_topo_feature|'recon'|\"recon\""` and list every hit outside `docs/`, `output/`, `CLAUDE.md`; every code hit must be either deleted by this task or unrelated (`recon` outputs of the pretrain model are unrelated).
- [ ] **Step 2: Make the edits above.** Preserve CRLF in `MeSAE.py` (and `plugin.py`/`base_checker.py` if CRLF); `git diff --stat` must show deletions, not whole-file rewrites.
- [ ] **Step 3: Verify.**
  - `python -c "import train_finetune, check_model, viz, model.factory"` (eeg_fm, CPU) imports cleanly.
  - `git grep -n -E "MeSAEFinetune|PerChannelHeadAttn|head_z|z_proj|use_topo_feature"` shows no hits outside `docs/`, `CLAUDE.md`, `CONTEXT.md` history text.
  - Re-run `head_equiv.py` from Task 2 with its old class taken from the tag (`git show pre-head-cleanup:model/MeSAE/MeSAE.py`), dropping the configurations for the removed options (`recon`, `z_chan`, `concat`): every remaining configuration and the real C1 checkpoint still print `OK`.
  - `viz.load_model` smoke check from Task 2 Step 5 still prints `MeSAEFeatureHead torch.Size([2, 25]) torch.Size([25])`.
  - Build the base config's head: `build_finetune_from_config` on `config/config.json` (a `stamp_induced` head, `num_classes=4`, `num_patches=39`) succeeds.
- [ ] **Step 4: Commit** in two commits: `refactor: remove the head_z path (MeSAEFinetune, PerChannelHeadAttn, render_finetune_attn)` and `refactor: drop recon, z_chan and channel-concat options from MeSAEFeatureHead; delete Experiment A probes`, each with the two attribution lines.

---

## Self-Review Notes

- **Coverage of ADR 0016:** modules 2 to 4 (spatial mix, time pooling, optional branches) are separate pieces; feature source stays inside `MeSAEFeatureHead` (its five inputs need the backbone and differ too much to share an interface, and ADR 0016 keeps them as an option list); dropout, K, R remain constructor arguments; readout untouched.
- **Interfaces:** `phase_advance` and `EvokedBranch.forward` both return `[B, K, 2*S]` in re/im order, matching the previous concatenation order, so `n_feat` (`K*S*(1+2*adv+2*evk)`) is unchanged.
- **Risks handled:** RNG order and state-dict identity are asserted before any forward pass; the gate on `ALLDONE` protects the running Phase 2 queue; the temp old-class file is never committed.
- **Deliberately not done:** no new ablation options (`window` and flat already exist) and no config-format change; the ablation grid itself is a separate plan once step 0 is done.
