# Experiment C — C4 (evoked branch) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the evoked branch (`2b` in ADR 0014's Head pseudocode) to
`MeSAEFeatureHead`'s `stamp_induced` arm, on top of **C1's** configuration (spatial
filter + per-stamp induced power + learned rank-2 time weights — the best head so far,
tail-mean 0.536), then run the wiring check ADR 0014 build-order step 12 calls for:
**C4 must beat C1**.

**Why C1 and not C3 as the base:** C3 (phase advance) regressed significantly against C1
(0.489 vs 0.536, p=0.010, build-order step 10). The ladder is cumulative only while each
step wins; C1 remains the best head, so C4 adds its branch to C1, not to C3. The C3 branch
stays in the code (`include_advance`, off by default) but is **off** for this run.

**Architecture:** Extend `MeSAEFeatureHead` again (same additive, opt-in pattern as C1's
`pool_time="learned:R"` and C3's `include_advance`) rather than adding a class. Add one new
constructor kwarg, `evoked_rank` (int, default `0` = off; `R > 0` = on with a rank-`R`
time filter), valid only for `input="stamp_induced"` with `pool_time` not a `window:` form.

The evoked term (ADR pseudocode `2b. evoked: sum_n T[s,n] u`) is a **signed, learned,
low-rank time filter applied to the complex code** `u = a + i·b` (the same `a, b` the
induced branch already computes, post spatial filter, shape `[B, N', K, S]`):

```
T[s,n]   = 1/N' + Σ_r p_r[s]·q_r[n]        # [S, N'], signed (NOT softmax-normalized)
ev_re[k,s] = Σ_n T[s,n]·a[n,k,s]            # [B, K, S]
ev_im[k,s] = Σ_n T[s,n]·b[n,k,s]            # [B, K, S]
```

Differences from C1's induced time weights, on purpose: (1) **signed** (an evoked response
is a signed waveform; a softmax would force a weighted mean and could not express a
"positive early, negative late" filter), (2) applied to `a` and `b` **linearly** (not to
`a²+b²`) — a linear functional of the code is what preserves phase-locked content, power
destroys it, (3) its **own** `p, q` parameters, independent of the induced branch's.
The `1/N'` offset makes the branch start as the plain trial-mean of `(a, b)` (the
time-locked average) and learn a deviation from it, mirroring C1's "start equivalent to the
flat baseline" design; `p, q` init `randn * 0.02` like C1.

`ev_re`/`ev_im` are fed raw (no log — they can be negative) into the same shared
`BatchNorm1d → Dropout → Linear` classifier, concatenated on the last axis after the
induced (and, if enabled, advance) features.

**Known, pre-declared expectation (not an oversight):**
- ADR 0014 itself calls this step "needed for ERP; likely neutral on MI" (ablation
  ladder), and the stamp-relevance analysis found inter-trial phase coherence at or below
  the bias floor for every stamp on BCICIV2a — i.e. no phase-locked stamps for MI.
- The branch adds `2·K·S` = 400 features → head input width `K·S·3` = **600**, the same
  regime C3 overfit in (train acc 0.9255, significantly worse validation). No group
  penalty is implemented (C0/C1/C3 precedent; `dropout: 0.5` alone).
- So the most likely outcome on BCICIV2a MI is **no gain or a C3-like regression** — a
  measurement of "the cost of 400 extra features carrying no MI signal," not of evoked
  information. That is an acceptable, honest result to record; it is *not* a reason to
  change dropout/epochs to rescue the number. **The real evaluation of `2b` is ERP
  (Inria) and SSVEP (BETA)** against their own raw baselines (build-order step 12,
  "then ERP on Inria…") — that needs an ERP task block the head does not have yet
  (`task="mi"` only) and is **out of scope for this plan**; record it as the next step.

**Tech Stack:** PyTorch (einsum on existing tensors, one `nn.ParameterDict`), the existing
`train_finetune.py` / `probes/ft_summary.py` pipeline (unmodified; `ft_summary.py` already
groups by subject before its paired t-test).

**Spec:** `docs/adr/0014-finetune-head-test-plan.md` — the `Head` pseudocode block
(`2b. evoked`), the "Ablation ladder" table (`C4 | evoked branch (2b) | needed for ERP;
likely neutral on MI`), the "Size control is mandatory" paragraph, the Protocol section's
acceptance-comparison methodology (tail-mean point comparison, n=9 paired t reported but
never a pass/fail gate), and build-order steps 8 (C1, the baseline), 10 (C3, why it is
not the base), 12 (C4 itself).

## Global Constraints

- Backbone stays frozen (ADR 0012) — do not touch `freeze_backbone`.
- Backbone checkpoint: `output/mesae_v10_small_uw01/checkpoint/last.pth`.
- Protocol (ADR 0014 §Protocol): `learning_rate: 0.01`, `min_learning_rate: 0.001`,
  `epochs: 100`, `warmup_epochs: 2`, `split_mode: "intra_subject_cv"`, `cv_folds: 5`.
- Head `dropout: 0.5` — unchanged from C1. Do NOT change dropout, weight decay, epochs, or
  add any size control to rescue a bad result; the run tests exactly one new factor.
- The C4 run's config must equal **C1's** config plus `evoked_rank: 2`: `input:
  "stamp_induced"`, `pool_channel: "spatial:8"`, `pool_time: "learned:2"`, `dropout: 0.5`,
  `task: "mi"`, `freeze_backbone: true`, and `include_advance` **absent or false** (the live
  `config/config.json` currently has `include_advance: true` from the C3 run — this must
  be removed/false for C4, otherwise C4 silently becomes C3+C4 = 800 extra features).
- Reporting metric: `balanced_acc`, mean over each subject's last 10 epochs
  (`probes/ft_summary.py` "tail" row).
- Acceptance comparison: tail-mean point difference vs C1 (0.536), paired t-test across the
  9 subjects (folds averaged within subject first — `ft_summary.py` does this), reported
  honestly regardless of significance.
- No project test suite exists (CLAUDE.md) — validation is empirical: a shape/build/grad
  smoke check for Task 1, the real training run for Task 2.
- Do not touch `model_params.MeSAE.pretrain.stamp_bank` (`60/4/12/6`) or
  `dataset_params.finetune`.
- All other inputs (`raw`, `recon`, `stamp_bandpow`, `z_chan`) and `stamp_induced` with
  `evoked_rank=0` must be completely unaffected — additive, opt-in.

---

## Task 1: `evoked_rank` evoked branch on `MeSAEFeatureHead`

**Files:**
- Modify: `model/MeSAE/MeSAE.py` — `MeSAEFeatureHead.__init__`, `.forward` (the
  `stamp_induced` branch), class docstring, and `build_finetune`'s `allowed` tuple.

**Interfaces:**
- Produces: `MeSAEFeatureHead(..., evoked_rank=0)` and `build_finetune(..., evoked_rank=0)`.
- Consumes: the existing `a, b` tensors in the `stamp_induced` branch of `forward`;
  `num_patches` (already threaded through `build_finetune_from_config` by the C1 plan).

- [ ] **Step 1: Read the live class**

Read `model/MeSAE/MeSAE.py`'s `MeSAEFeatureHead` (search `class MeSAEFeatureHead`) top to
bottom. Confirm: the `include_advance` guard block right after the `input not in (...)`
check; the `head['time'] = nn.ParameterDict({...})` block (registered **before**
`self.head = nn.ModuleDict(head)`); the `n_feat` line for `stamp_induced`
(`K * len(keep) * (3 if include_advance else 1)`); the `if self.include_advance:` block at
the end of the `stamp_induced` branch of `forward`; and `build_finetune`'s `allowed` tuple.
Line numbers are anchors only — match the live file.

- [ ] **Step 2: Constructor kwarg, guards, `self.evoked_rank`**

Add `evoked_rank=0` after `include_advance=False` in the signature:

```python
                 include_advance=False, evoked_rank=0):
```

Right after the existing `include_advance` guard block (and its `self.include_advance = ...`
line), add:

```python
        if evoked_rank and input != 'stamp_induced':
            raise NotImplementedError(
                f"evoked_rank is only implemented for input='stamp_induced' "
                f"(ADR 0014 experiment C4), got input={input!r}")
        if evoked_rank and num_patches is None:
            raise ValueError("evoked_rank requires num_patches (pass it through "
                              "build_finetune_from_config -- see model/factory.py)")
        if evoked_rank and pool_time.startswith('window:'):
            raise NotImplementedError(
                "evoked_rank needs the full patch axis (N' == num_patches); it cannot be "
                "combined with a window: pool_time, which slices patches")
        self.evoked_rank = int(evoked_rank)
```

- [ ] **Step 3: Register the time-filter parameters (before `nn.ModuleDict(head)`)**

Immediately after the existing `if self.time_rank is not None: ... head['time'] = ...`
block (and still before `if input == 'z_chan':`), add:

```python
        if self.evoked_rank:
            # Signed low-rank time filter for the evoked branch (2b), T[s,n] = 1/N' +
            # sum_r p_r[s] q_r[n]. Small random p,q => starts as the plain trial-mean of (a,b)
            # (the time-locked average) and learns a deviation. nn.ModuleDict rejects raw
            # nn.Parameter values (same constraint as head['time']), hence the ParameterDict.
            head['evoked'] = nn.ParameterDict({
                'p': nn.Parameter(torch.randn(self.evoked_rank, len(keep)) * 0.02),
                'q': nn.Parameter(torch.randn(self.evoked_rank, num_patches) * 0.02),
            })
```

(`keep` is already in scope — computed near the top of `__init__`.)

- [ ] **Step 4: Widen `n_feat`**

Change the `stamp_induced` `n_feat` line to:

```python
            n_feat = K * len(keep) * (1 + 2 * bool(include_advance) + 2 * bool(self.evoked_rank))
```

(1 induced block; +2 blocks for advance re/im; +2 blocks for evoked re/im. C4 alone: 3x.)

- [ ] **Step 5: Compute and concatenate the evoked features in `forward`**

Inside the `stamp_induced` branch, immediately **after** the closing of the
`if self.include_advance:` block (same indentation as that `if`), add:

```python
                if self.evoked_rank:
                    # C4 (ADR 0014 build-order step 12): evoked branch (2b),
                    # sum_n T[s,n] * u[n,k,s], u = a + i*b. T is signed and applied to a and b
                    # LINEARLY (not to power) -- a linear functional preserves phase-locked
                    # content, power destroys it. Same a, b as above, no new backbone call.
                    # Fed raw (can be negative); the shared BatchNorm1d normalizes scale.
                    T = 1.0 / a.shape[1] + torch.einsum(
                        'rs,rn->sn', self.head['evoked']['p'], self.head['evoked']['q'])   # [S, N']
                    ev_re = torch.einsum('sn,bnks->bks', T, a)                             # [B, K, S]
                    ev_im = torch.einsum('sn,bnks->bks', T, b)                             # [B, K, S]
                    feat = torch.cat([feat, ev_re, ev_im], dim=-1)
```

This sits inside the existing `torch.autocast(..., enabled=False)` block and after the
`torch.no_grad()` block has exited (same placement as the C1/C3 math), so gradients reach
`head['evoked']['p'/'q']` and the spatial filter.

- [ ] **Step 6: `allowed` tuple and docstring**

`build_finetune`'s non-`head_z` `allowed` tuple: append `'evoked_rank'`:

```python
    allowed = ('task', 'pool_channel', 'pool_time', 'z_proj', 'dropout', 'sample_freq',
               'freeze_backbone', 'num_patches', 'include_advance', 'evoked_rank')
```

Add to the class docstring, after the `include_advance:` entry:

```python
    evoked_rank:  int, stamp_induced only, 0 = off (ADR 0014 C4) -- adds the evoked branch
                  (2b): a signed rank-R time filter T[s,n] applied linearly to the complex
                  code (a, b), giving re/im features (+2*K*S width, K*S -> K*S*3 alone).
                  Needs num_patches and the full patch axis (no window: pool_time). Like
                  learned:R and include_advance, depends on fixed-length trials.
```

- [ ] **Step 7: Smoke check (no training run)**

```bash
python -c "
import json, torch
from model.factory import build_pretrain_from_config
from model.MeSAE.MeSAE import build_finetune

config = json.load(open('config/config.json'))
backbone = build_pretrain_from_config(config, mode='finetune'); backbone.eval()
backbone.stamps.fire_ema.fill_(1.0)   # force a non-empty alive set on a fresh backbone
C, n_patches = 64, 30                  # canonical_channels='10-10'; synthetic patch count
kw = dict(input='stamp_induced', pool_channel='spatial:8', pool_time='learned:2', task='mi',
          sample_freq=config['preprocess_params']['sample_freq'], num_patches=n_patches, dropout=0.5)
B, L = 2, config['preprocess_params']['patch_length']
x = torch.randn(B, C, n_patches, L); coords = torch.randn(B, C, 3)
ti = torch.arange(n_patches).unsqueeze(0).expand(B, n_patches)
vc = torch.ones(B, C, dtype=torch.bool)

m = build_finetune(backbone, C, 4, evoked_rank=2, **kw)
S = m.keep.numel()
lg, _, _ = m(x, coords, time_idx=ti, valid_channels=vc)
assert lg.shape == (B, 4)
assert m.head['cls'][2].in_features == 8 * S * 3, m.head['cls'][2].in_features
assert m.head['evoked']['p'].shape == (2, S) and m.head['evoked']['q'].shape == (2, n_patches)
assert any(n.startswith('evoked.') for n, _ in m.head.named_parameters()), 'evoked params not in head.parameters()'
lg.sum().backward()
for nm in ('p', 'q'):
    g = m.head['evoked'][nm].grad
    assert g is not None and g.abs().sum() > 0, f'no grad on evoked.{nm}'
print('evoked: shapes, width, registration, gradient OK')

# init sanity: with p,q ~ 0 the evoked branch equals the plain trial-mean of (a,b)
assert m.head['evoked']['p'].abs().max() < 0.2

m0 = build_finetune(backbone, C, 4, evoked_rank=0, **kw)      # C1 config regression
assert m0.head['cls'][2].in_features == 8 * S and 'evoked' not in m0.head
lg0, _, _ = m0(x, coords, time_idx=ti, valid_channels=vc); assert lg0.shape == (B, 4)
print('evoked_rank=0 regression: OK')

m2 = build_finetune(backbone, C, 4, evoked_rank=2, include_advance=True, **kw)
assert m2.head['cls'][2].in_features == 8 * S * 5           # induced + advance(2) + evoked(2)
print('advance+evoked width 5x: OK')

for bad, exc in [
    (dict(input='stamp_bandpow', evoked_rank=2), NotImplementedError),
    (dict(evoked_rank=2, num_patches=None), ValueError),
    (dict(evoked_rank=2, pool_time='window:0.5-2.5'), NotImplementedError),
]:
    try:
        build_finetune(backbone, C, 4, **{**kw, **bad}); raise SystemExit(f'expected {exc.__name__}: {bad}')
    except exc: pass
print('guards: OK')
print('ALL OK')
"
```

Expected: no exception; `ALL OK` last. If the `a.shape[1] != num_patches` mismatch shows up
(a shape error in `torch.einsum('sn,bnks->bks', T, a)`), that means the check's
`n_patches` differs from the `N` fed to the forward pass — fix the script, not the model.

- [ ] **Step 8: Commit**

```bash
git add model/MeSAE/MeSAE.py
git commit -m "$(cat <<'EOF'
feat: add evoked_rank evoked branch to MeSAEFeatureHead (ADR 0014 C4)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011ZRo7FMWpgbtn1N34keF18
EOF
)"
```

---

## Task 2: Wire config, run C4, update the ADR

**Files:**
- Modify: `config/config.json` (`model_params.MeSAE.finetune`, `training_params.finetune`)
- Modify: `docs/adr/0014-finetune-head-test-plan.md` (build-order step 12)

- [ ] **Step 1: Point the finetune config at C4 (= C1 + `evoked_rank: 2`)**

Set `model_params.MeSAE.finetune` to exactly:

```json
      "finetune": {
        "input": "stamp_induced",
        "task": "mi",
        "pool_channel": "spatial:8",
        "pool_time": "learned:2",
        "dropout": 0.5,
        "freeze_backbone": true,
        "evoked_rank": 2
      }
```

This **removes `include_advance`** (present and `true` from the C3 run). Set
`training_params.finetune.model_name` to `"mesae_finetune_c4_evoked"`; every other field
stays as it is (it already equals C1's protocol: 5-fold, lr 0.01/0.001, 100 epochs, warmup
2, checkpoint `mesae_v10_small_uw01`). Verify against
`output/experiment_c/mesae_finetune_c1_learned2/artifacts/config.json` (the config C1 actually ran with)
that the only differences are `evoked_rank: 2` and `model_name`.

- [ ] **Step 2: Run C4**

```bash
python train_finetune.py --config config/config.json
```

Real GPU job, 9 subjects × 5 folds × 100 epochs (~3h40m as C1/C3). Let it finish. Then:
`python probes/ft_summary.py output/experiment_c/mesae_finetune_c1_learned2/artifacts/train_20260919_192402.log output/experiment_c/mesae_finetune_c4_evoked/artifacts/train_<timestamp>.log`
(C1 first so the printed paired line is C4 − C1; the script groups by subject).
Extract final-epoch train `balanced_acc` over the 45 runs (e.g. `grep "\[Train\]" <log> | awk` every 100th, or a one-off script) and the trainable head parameter count from a saved
checkpoint (`sum(v.numel() for k,v in sd.items() if k.startswith('head.'))` excluding BN
running buffers, as the C3 entry did: C1 1,844; C3 4,244).

On a crash for an environment/config reason: diagnose and report; do not change what is
measured. A narrow new code bug fix is allowed (own commit, documented deviation).

- [ ] **Step 3: Update the ADR**

Extend build-order step 12's C4 entry (`12. **C4 — evoked branch (2b)**, then ERP on Inria
and SSVEP on BETA…`) in the ADR's existing prose style (bolded lead phrases, inline
numbers, no new tables), reporting: tail-mean vs C1's 0.536, per-subject tail means, the
`ft_summary.py` n=9 paired stats (mean diff, t, p, wins) honestly regardless of
significance; final-epoch train accuracy vs C1's 0.843 / C3's 0.9255 / C0-family
~0.74 / unregularized ~0.985; head parameter count vs C1 1,844 / C3 4,244 / the ~1k target;
and a plain, evidence-only reading (no gain / regression / gain), stating that this run
measures `2b` **on MI, where the ADR predicted it neutral and found no phase-locked
stamps**, so it is not the evoked branch's real test. Name the real next step: ERP (Inria)
and SSVEP (BETA) against their own raw baselines, which needs an ERP task block the head
lacks (`task="mi"` only). Do NOT mark step 12 done. Do not modify steps 7–10 or the
Protocol section.

- [ ] **Step 4: Commits**

Two commits: config (before running), ADR (after). Match `git log -5` style and end with
the `Co-Authored-By` + `Claude-Session` trailers used in Task 1's commit.

## Self-Review Notes

- **Spec coverage:** ladder row C4 / pseudocode `2b` → Task 1 (branch) + Task 2 (run and
  record). ERP/SSVEP evaluation is explicitly out of scope and named as the next step.
- **Placeholders:** none; all code and commands literal.
- **Type consistency:** `evoked_rank` (Task 1) is exactly the key Task 2's config sets;
  `head['evoked']['p'|'q']` names match between `__init__` and `forward`; `n_feat` factor
  `1 + 2·advance + 2·evoked` matches the concatenation order (induced, advance, evoked)
  and the smoke check's asserted widths (3x alone, 5x with both).
- **Pitfalls carried from C1/C3:** `nn.ModuleDict` rejects raw `nn.Parameter` (used
  `nn.ParameterDict`); params registered before `self.head = nn.ModuleDict(head)`; math
  placed after the `no_grad` block, inside the fp32 autocast block; `include_advance` must
  be off in the C4 config.
