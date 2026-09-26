# Rotary position embeddings for a 1M-parameter GPT trained on 9.8M targets

DASE7506 MP1 — small language model challenge

**Frozen test BPB: 1.9527** (supplied baseline reproduced at 2.1027, −0.1500 BPB, −7.13%)

---

## 1. Diagnosis and method

### 1.1 The regime

The task fixes the optimisation budget at `steps × batch × context = 1200 × 32 × 256 = 9,830,400`
processed targets, against a model of ~1.06–1.09M parameters. That is roughly **nine training
tokens per parameter** — two to three orders of magnitude below the compute-optimal ratio for a
transformer language model. The regime is therefore **compute-limited, not capacity-limited**, and
the validation curves confirm it rather than assuming it: every arm in this study is still
improving when training stops (Table 1). Nothing here is at risk of overfitting; the question is
only which architecture extracts the most from a fixed, small number of gradient steps.

**Table 1 — validation BPB during training, seed 17** (`--eval-every 300`)

| arm | 300 | 600 | 900 | 1200 |
|---|---:|---:|---:|---:|
| `learned` (baseline) | 2.3323 | 2.1849 | 2.1074 | 2.0729 |
| `none` | 2.3080 | 2.1816 | 2.1357 | 2.1148 |
| `rope` | 2.2262 | 2.0320 | 1.9547 | 1.9222 |

No curve flattens. Under a budget this tight, an *inductive bias* that removes work the model would
otherwise have to discover from data is likely to pay more than additional parameters would.

### 1.2 The limitation

The baseline injects position as a learned absolute embedding added to the token embedding:

```python
x = self.token(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device))
```

`self.pos` is a 256×128 table — 32,768 parameters, 3.0% of the model. This encodes *where* a token
is, absolutely. But attention fundamentally wants *how far apart* two tokens are: the pattern
"attend to the previous token" is a fixed offset, and under absolute position embeddings the model
must learn that offset separately for each of the 256 starting positions, or learn to subtract two
absolute codes. With 1,200 optimiser steps there is little opportunity to discover that structure.

The method replaces the table with **rotary position embeddings (RoPE)**, applied to queries and
keys inside every attention block. Position enters multiplicatively as a rotation by an angle
proportional to the absolute index; because attention scores depend on the *difference* of two
rotations, the resulting scores depend only on the **relative** offset `i − j`. Relative structure
is built into the parameterisation instead of being learned. The table disappears entirely.

### 1.3 Why this is the right test

Replacing `learned` with `rope` changes two things at once: the position *scheme*, and the parameter
*count* (32,768 fewer). To separate them the study includes a third arm, `none`, which applies **no
position information at all** — no table, no rotation, nothing. It has parameters **identical to
`rope`** (1,055,488). This is what makes the comparison interpretable, and it is the reason the
ablation has three arms rather than two:

| comparison | parameters | difference | isolates |
|---|---|---:|---|
| `rope` vs `none` | identical | **+0.194 BPB** | relative-position structure, with capacity held fixed |
| `learned` vs `none` | +32,768 | +0.040 BPB | what a learned absolute table buys |
| `learned` vs `rope` | +32,768 | +0.154 BPB | confounded — reported for completeness only |

## 2. Experimental setup

- **Data / protocol**: WikiText-2 raw, train-fitted BPE-2048, context 256, protocol `7506-mp1-wt2-v2`.
  Data, tokenizer and evaluator were not modified; `data/manifest.json` hashes verified on every run.
- **Arms**: `learned` / `none` / `rope`, identical in every other respect — same config file apart
  from the `pos_mode` key, same depth 4, width 128, 4 heads, same step count, same batch size.
- **Seeds**: 3 per arm (17, 18, 19). All selection was done on validation.
- **Hardware**: RTX 3060 Laptop (6 GB), 16 logical cores. Training in BF16 on CUDA; every
  *reported* timing and memory figure is a separate **CPU FP32** pass, because that is the device the
  evaluation budget is defined on.
- **Baseline arm**: the supplied `model.py` via `--implementation model`, not a reimplementation.

### 2.1 Establishing that the ablation is clean

Two checks, both run before any result was interpreted (`rope.py`, `student.py`):

1. The `learned` arm is **bitwise identical** to `model.py`'s GPT: identical `state_dict` keys, and
   after loading the baseline's weights, identical outputs on a length-64 batch. So the ablation
   changes only how position reaches attention.
2. Parameter counts match exactly: `learned` = 1,088,256 (identical to `model.py`), `none` = `rope`
   = 1,055,488.

End-to-end confirmation: `learned`-s17 and the supplied baseline both score **2.0729** validation
BPB — identical to four decimals.

## 3. Results

**Table 2 — validation BPB, 3 seeds per arm, 9,830,400 targets each**

| arm | parameters | seed 17 | seed 18 | seed 19 | **mean ± std** |
|---|---:|---:|---:|---:|---:|
| `learned` (baseline) | 1,088,256 | 2.0729 | 2.0732 | 2.0783 | **2.0748 ± 0.0030** |
| `none` | 1,055,488 | 2.1148 | 2.1151 | 2.1144 | **2.1148 ± 0.0003** |
| **`rope`** | 1,055,488 | 1.9222 | 1.9235 | 1.9179 | **1.9212 ± 0.0029** |

The supplied baseline reproduced at **2.1027** on test (documented ≈2.10).

**Seed variance is not the explanation.** The `rope` advantage over `learned` is 0.1536 BPB against
a pooled seed standard deviation of 0.003 — a separation of roughly **50 standard deviations**. The
per-seed ranges do not overlap: `rope` spans 1.9179–1.9235, `learned` spans 2.0729–2.0783.

**Table 3 — frozen test score** (checkpoint SHA256 `b3efade1…`; see §7 on test-set usage)

| | test BPB | CPU FP32 scoring |
|---|---:|---:|
| supplied baseline (SHA256 `1fa27eac…`) | 2.1027 | 10.51 s |
| **`rope`, seed 17 (submitted)** | **1.9527** | 11.89 s |
| difference | **−0.1500 (−7.13%)** | 1.13× |

## 4. Ablation and analysis

### 4.1 The result is about relative structure, not about removing parameters

`none` and `rope` are the same size. They differ by 0.194 BPB. Whatever `rope` contributes, it is
not the 32,768 parameters it removes — capacity is held exactly fixed across that comparison. The
gain must come from the parameterisation itself.

### 4.2 Absolute position information is nearly worthless here

`learned` beats `none` by only 0.040 BPB, and it does so while carrying 32,768 *extra* parameters.
So the entire contribution of a learned absolute position table — the thing the baseline spends 3%
of its parameters on — is worth 0.04 BPB, less than a third of what the relative parameterisation
buys at equal cost.

### 4.3 Interpretation

The two findings combine into a consistent story. In a 1,200-step budget the model cannot afford to
*discover* relative structure from data. Supplying relative offsets directly in the attention
parameterisation does most of the work; supplying absolute positions and hoping the model learns to
subtract them does almost nothing. This is exactly the regime where an inductive bias should beat
learned capacity, and the measurements bear it out.

A secondary effect is plausible but **not isolated by this experiment**: the rotation is a
fixed, norm-preserving transformation, which may improve conditioning of the attention logits during
early training. Separating that from the relative-offset story would require an arm with relative
structure but no rotation — see §6.

## 5. Cost

All figures from a separate CPU FP32 pass over the frozen checkpoints, `--threads 8`, on the same
predictor used for the reported scores.

**Table 4 — resource budgets**

| budget | baseline | `rope` | ratio | limit | |
|---|---:|---:|---:|---:|---|
| CPU scoring, test | 10.51 s | 11.89 s | **1.13×** | ≤ 5× | pass |
| CPU scoring, validation | 9.21 s | 10.77 s | 1.17× | — | |
| Peak RAM | 1.784 GiB | 1.785 GiB | — | ≤ 4 GiB | pass |
| Inference assets | 4.17 MiB | 4.04 MiB | — | ≤ 64 MiB | pass |

**Prediction quality against cost.** `rope` costs **+13% CPU scoring time and 21% more training
time** (28.3 s vs 23.3 s per 1,200 steps, mean of 3 seeds) in exchange for **−0.150 test BPB**. It is also *smaller*
on disk (4.04 vs 4.17 MiB) and in parameters. The whole ablation matrix — ten 1,200-step runs —
cost 4.25 minutes of GPU time (6.95 minutes wall clock, including per-run data preparation), so the
search cost is negligible against the deadline.

The extra scoring time is worth stating plainly: `apply_rope` adds elementwise work on queries and
keys at every block, and at width 128 with 4 layers that overhead is a larger fraction of the
forward pass than the 32,768 saved parameters are. So this is *not* a free improvement — it trades
13% of the scoring budget for 7.1% of the error. With a 5× budget available, that is a favourable
trade, but it is a trade.

**Training cost disclosure.** Every run is 1,200 steps × 32 × 256 = 9,830,400 targets from random
initialisation, with no parent checkpoints, so no ancestry needs to be accumulated. Ten runs total:
4.25 minutes of GPU training in aggregate (sum of `train_seconds`), 6.95 minutes wall clock
including data preparation and evaluation. Full per-run accounting is in `RUN_LOG.csv`.

## 6. Critical analysis

**What this experiment does establish.** At 1M parameters and 9.8M targets, replacing a learned
absolute position table with rotary embeddings reduces test BPB by 0.150 (7.1%), with the effect
about 50× the seed noise, at equal or lower parameter count, within all three resource budgets.

**What it does not establish.**

- **Rotation vs relative structure.** RoPE is one way to obtain relative offsets. The `none` arm
  controls for parameter count, but nothing here separates "relative offsets help" from "this
  particular rotation helps" — a fixed relative bias (e.g. ALiBi) or a learned relative bias would
  be the natural next arms. Given the result, I would expect a fixed additive bias to capture much
  of the same gain; **that comparison is the single most informative experiment this study omits.**
- **Scale.** One architecture family (width 128, depth 4), one context length (256), one dataset.
  The compute-limited regime is the stated reason the result should generalise, but that is an
  argument, not a measurement. I would not extrapolate to a model trained to convergence.
- **Three seeds.** The effect is far outside the seed spread observed here, but three seeds is a
  small sample for a standard deviation; the ±0.003 figures should be read as a rough scale, not a
  precise estimate.
- **No hyperparameter search.** `base = 10000` and the half-split frequency layout were taken as
  standard and never tuned. The reported number is the result of one design decision, not of a
  search — which makes the comparison cleaner but the absolute score almost certainly beatable.
- **Confounded capacity in two of three comparisons.** Only `rope` vs `none` holds parameters fixed.
  The `learned` vs `rope` difference of 0.154 should not be read as a pure method effect.

**Directions this result suggests but does not test.** Because the regime is compute-limited
(Table 1: no curve has flattened), the largest remaining gains are probably in *optimisation
efficiency* rather than architecture. The supplied recipe uses AdamW at lr 1e-3 with 100 warmup
steps into cosine decay, no weight decay exemption for norms or biases, and no weight averaging.
Weight averaging (EMA) is cheap, almost always helps, and would compose with the change here. That
is where I would look next, and I did not test it — a deliberate choice to keep the ablation
single-variable inside a two-day budget.

**On seed selection.** The submitted checkpoint is **seed 17**, which is `train.py`'s default and the
seed used throughout the supplied README examples — it was fixed before the sweep, not chosen from
it. Seed 19 happened to score lower (1.9179) and was *not* used.

## 7. Reproduction

```bash
cd MP1_student_starter/code
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v                # contract gate
```

```bash
# All ten runs (~4.5 min on an RTX 3060); writes code/runs/*/metrics.json
bash ../run_matrix.sh

# Reproduce the reported score from the submitted checkpoint, no retraining
python evaluate.py --checkpoint runs/rope-s17/checkpoint.pt \
  --device cpu --precision fp32 --threads 8 --split test

# Budget measurements
python ../measure_peak_ram.py runs/rope-s17/checkpoint.pt validation
python ../make_run_log.py                              # regenerate RUN_LOG.csv
```

The evaluator writes `test_cpu_fp32.json` beside the checkpoint. `bpb` in that file is the score
reported above. Timing figures in §5 use `--threads 8`; the score itself is thread-independent.

**Layout of the method.** `code/rope.py` (the rotation itself) and `code/student.py` (the model,
with `config['pos_mode']` selecting the arm) are the only files that differ from the baseline;
`code/model.py`, `code/common.py` and `code/evaluate.py` are untouched. `code/configs/pos-*.json`
mirror `baseline.json` with one extra key, which `model.py` ignores.

**Training is bit-reproducible here.** Re-running `rope` at seed 17 from scratch — same config, same
commit, same machine — reproduces the checkpoint **byte for byte** (identical SHA256 `b3efade1…`)
and therefore the same validation BPB to four decimals. This is a property of this configuration on
this hardware, not a promise that BF16 CUDA training is reproducible in general.

**Reusing a checkpoint does not erase its training cost.** All figures above are for models trained
from random initialisation for the full 9,830,400 targets; nothing is fine-tuned from an ancestor.

### Acknowledgement of AI assistance

Claude (Anthropic) was used as a coding and analysis assistant for this coursework. Specifically:
drafting `rope.py` and the `pos_mode` switch in `student.py`; writing the experiment driver
(`run_matrix.sh`), the peak-RAM probe (`measure_peak_ram.py`) and the log generator
(`make_run_log.py`); and drafting this report. All experimental design decisions, the choice of
RoPE over alternatives, the three-arm ablation structure, the seed-selection policy and the
interpretation in §4 and §6 were reviewed and are the author's responsibility. Every number
reported here was produced by the supplied harness on the hardware described in §2 and can be
regenerated with the commands in §7. The author has verified that the implementation is causal and
contract-test compliant, that the `learned` arm is bitwise identical to the supplied baseline, and
that no test-set information influenced any development decision.

**On test-set usage.** All development and selection was done on validation. The test split was
scored exactly twice, after the method was frozen: once for the submitted method and once for the
baseline arm, the latter being the workflow the supplied README documents for the baseline (§2). No
result from either pass changed any configuration, hyperparameter or checkpoint choice — the frozen
model had already been trained and selected on validation before the first test score was computed.
A third scoring pass was run to reproduce the submitted number from a fresh clone. The same frozen
predictor may be re-scored freely; that is what the reproduction instructions in §7 invite.

### References

- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, arXiv:2104.09864.
- Merity et al., *Pointer Sentinel Mixture Models*, arXiv:1609.07843 (WikiText-2).
- Loshchilov & Hutter, *Decoupled Weight Decay Regularization*, arXiv:1711.05101 (AdamW).
