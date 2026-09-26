# Rotary position embeddings, and what a tiny compute budget does to their value

DASE7506 MP1 — small language model challenge

**Frozen test BPB: 1.7228** · initial baseline 2.1027 · matched-target baseline 1.7788
(−18.1% against the initial baseline, −0.0560 against the same-targets baseline)

---

## 1. Diagnosis and method

### 1.1 The regime

The supplied recipe processes `1200 × 32 × 256 = 9,830,400` targets against ~1.06–1.09M parameters —
about **nine training tokens per parameter**, two to three orders of magnitude below the
compute-optimal ratio. The regime is **compute-limited, not capacity-limited**, and the validation
curves say so rather than assuming it: at the baseline's 1,200 steps no arm is close to flattening
(Table 1).

**Table 1 — validation BPB during training, seed 17**

| arm | 300 | 600 | 1200 | 2400 | 3600 | 4800 |
|---|---:|---:|---:|---:|---:|---:|
| `learned` | 2.3323 | 2.1849 | 2.0729 | — | — | 1.7532 |
| `none` | 2.3080 | 2.1816 | 2.1148 | — | — | 1.8314 |
| `rope` | 2.2262 | 2.0320 | 1.9222 | 1.7575 | 1.7120 | 1.6945 |

Under a budget this tight there are two candidate levers: give the model a better *inductive bias*
so it needs fewer steps to reach the same place, or simply **give it more steps**. This study tests
both, and the honest summary is that the second moved the score far more than the first. The
mechanism under study is the first; §4 and §5 report the second as a cost decision, not as a
contribution.

### 1.2 The mechanism

The baseline injects position as a learned absolute embedding:

```python
x = self.token(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device))
```

`self.pos` is a 256×128 table — 32,768 parameters, 3.0% of the model — encoding *where* a token is.
Attention wants *how far apart* two tokens are: "attend to the previous token" is a fixed offset,
and under absolute position embeddings the model must learn that offset at each of 256 positions, or
learn to subtract two absolute codes. The method replaces the table with **rotary position embeddings
(RoPE)**: queries and keys are rotated by an angle proportional to their absolute index, so attention
scores depend only on the **relative** offset `i − j`. Relative structure is built into the
parameterisation instead of being discovered from ~10M tokens.

### 1.3 Why the ablation has three arms, not two

Replacing `learned` with `rope` changes the position *scheme* and the parameter *count* at once. The
third arm, `none`, applies **no position information at all** — it has parameters **identical to
`rope`** (1,055,488). That is what makes the comparison interpretable:

| comparison | parameters | isolates |
|---|---|---|
| `rope` vs `none` | identical | relative-position structure, capacity held fixed |
| `learned` vs `none` | +32,768 | what a learned absolute table buys |
| `learned` vs `rope` | +32,768 | confounded — reported for completeness only |

## 2. Experimental setup

- **Data / protocol**: WikiText-2 raw, train-fitted BPE-2048, context 256, protocol `7506-mp1-wt2-v2`.
  Data, tokenizer and evaluator unmodified; `data/manifest.json` hashes checked on every run.
- **Arms**: `learned` / `none` / `rope`, identical apart from the `pos_mode` key in the config.
- **Seeds**: 3 per arm (17, 18, 19). All selection on validation.
- **Training lengths**: 1,200 steps (9,830,400 targets) and 4,800 steps (39,321,600 targets).
- **Hardware**: RTX 3060 Laptop (6 GB), 16 logical cores. Training in BF16 on CUDA; every *reported*
  timing and memory figure is a separate **CPU FP32** pass, the device the budget is defined on.
- **Baseline arm**: the supplied `model.py` via `--implementation model`, not a reimplementation.

### 2.1 Establishing that the ablation is clean

Both checks ran before any result was interpreted:

1. The `learned` arm is **bitwise identical** to `model.py`'s GPT — identical `state_dict` keys, and
   identical outputs after loading the baseline's weights.
2. Parameter counts: `learned` = 1,088,256 (identical to `model.py`), `none` = `rope` = 1,055,488.

End-to-end confirmation at *both* training lengths: `learned`-s17 and the supplied baseline both score
2.0729 (1,200 steps) and 1.7532 (4,800 steps) validation BPB — identical to four decimals.

## 3. Results

**Table 2 — validation BPB, 3 seeds per arm**

| arm | parameters | @1,200 mean ± std | @4,800 mean ± std |
|---|---:|---:|---:|
| `learned` (baseline) | 1,088,256 | 2.0748 ± 0.0030 | 1.7478 ± 0.0056 |
| `none` | 1,055,488 | 2.1148 ± 0.0003 | 1.8335 ± 0.0038 |
| **`rope`** | 1,055,488 | 1.9212 ± 0.0029 | **1.7022 ± 0.0074** |

Per-seed values are in `RUN_LOG.csv`. The submitted checkpoint is `rope`, seed 17 — `train.py`'s
default seed, fixed before the sweep — at 4,800 steps.

**Table 3 — frozen test scores** (CPU FP32, `checkpoints/`)

| | targets | test BPB | checkpoint SHA256 |
|---|---:|---:|---|
| initial baseline (`model.py`, 1,200 steps) | 9,830,400 | 2.1027 | `1fa27eac…` |
| matched-target baseline (`model.py`, 4,800) | 39,321,600 | 1.7788 | `6c7208a6…` |
| **submitted: `rope`, 4,800 steps** | 39,321,600 | **1.7228** | `128b8c0d…` |

Against the initial baseline: **−0.3799 BPB, −18.1%**. Against the same-targets baseline:
**−0.0560 BPB**.

## 4. Ablation and analysis

### 4.1 The mechanism works, and capacity is not the explanation

At 4,800 steps `rope` beats the parameter-matched `none` arm by **0.1313 BPB**. Since those two arms
are the same size, the gain cannot be the 32,768 parameters RoPE removes — it is the
parameterisation. `rope` also beats `learned` by **0.0456**, about six times the `rope` seed standard
deviation (0.0074).

### 4.2 The interesting result: the two gaps move in opposite directions

| gap | @1,200 steps | @4,800 steps | |
|---|---:|---:|---|
| `learned − none` (value of position information) | +0.0400 | **+0.0857** | ↑ grows |
| `learned − rope` (extra value of *relative* position) | +0.1536 | **+0.0456** | ↓ shrinks |

Per seed, the `rope` advantage over `learned` falls from 0.1507 / 0.1497 / 0.1604 to
0.0587 / 0.0389 / 0.0393 — **all three shrink**, so this is not seed noise.

The reading is consistent and, I think, the main scientific content of this study:

- **Position information of any kind becomes more valuable as training lengthens** (+0.040 → +0.086).
  A model with more optimisation steps can actually exploit position; one that is starved cannot.
- **The *extra* value of encoding position relatively rather than absolutely shrinks with training**
  (+0.154 → +0.046), because a longer-trained model can partly learn the relative structure that
  absolute embeddings only imply.
- **The two curves do not cross.** At 4,800 steps RoPE is still the best arm, so this is a narrowing
  advantage, not a vanished one. But a reader should take the 1,200-step figure of 0.154 as an upper
  bound that is specific to a severely under-trained regime — not as the durable size of the effect.

This is the clearest thing the ablation buys, and it required running the sweep at two lengths rather
than one.

### 4.3 Training longer dominates, and it is not a contribution

The largest single movement in this submission is not architectural. Going from 1,200 to 4,800 steps
takes the submitted model from 1.9527 to 1.7228 test BPB — **−0.230**, four times the size of the
RoPE effect at matched targets. Training duration is unrestricted by the assignment, training cost is
not part of the evaluation budget, and the required matched-targets comparison is satisfied by
Table 3. But it is a **cost decision, not an algorithmic contribution**, and it is reported as such.

**Why 4,800 and not more.** The per-1,200-step improvement collapses: 0.111 (1,200→2,400), 0.046
(2,400→3,600), 0.018 (3,600→4,800). Extrapolating that ratio, doubling again to 9,600 steps would buy
roughly 0.012 BPB — for a full re-run of the nine-run matrix. 4,800 is where the curve stops paying.

## 5. Cost

Separate CPU FP32 passes over the frozen checkpoints, `--threads 8`, same predictor as the reported
scores.

**Table 4 — resource budgets**

| budget | baseline | `rope` @4,800 | ratio | limit | |
|---|---:|---:|---:|---:|---|
| CPU scoring, validation | 6.71 s | 7.87 s | **1.17×** | ≤ 5× | pass |
| CPU scoring, test | 10.47 s | 11.93 s | 1.14× | — | |
| Peak RAM | — | 1.737 GiB | — | ≤ 4 GiB | pass |
| Inference assets | 4.18 MiB | 4.05 MiB | — | ≤ 64 MiB | pass |

**Quality against cost.** RoPE buys −0.056 test BPB at matched targets for **+17% CPU scoring time
and +14% training time** (108 s vs 95 s per 1,200 steps), while being *smaller* on disk and in
parameters. The scoring overhead is not free: `apply_rope` adds elementwise work on queries and keys
at every block, which at width 128 is a larger fraction of the forward pass than the 32,768 saved
parameters are. With a 5× budget available this is a favourable trade, but it is a trade.

**Training cost disclosure.** Every run is from random initialisation with no parent checkpoints, so
no ancestry accumulates. **24 training runs are logged in `RUN_LOG.csv`**, including the four
reproduction probes of §7; GPU training time sums to **27.2 minutes** (1,200-step runs ≈ 23–31 s
each, 4,800-step ≈ 95–108 s). The whole search cost is under half an hour on one consumer GPU.

### 5.1 Two things I tried that did not work

Both were measured, both are negative results, and both are recorded because the report's cost
argument depends on them.

- **Ensembling the three seeds.** Averaging the predictive distributions of the three `rope`@4,800
  checkpoints gives 1.6288 validation, a genuine −0.073 over the best single member — but it costs
  **33.2 s against a 6.7 s baseline, i.e. 5.01×**, exactly on the 5× line. The ensemble's cost is not
  3× the single model's: it is memory-bandwidth-bound, because it materialises an `[N, B, T, 2048]`
  log-probability tensor. A budget compliance claim resting on a 0.2% margin, on hardware that may
  differ from mine, is not a claim worth making. Rejected.
- **Averaging the three seeds' weights.** Catastrophic: 3.3122 validation, far worse than any member.
  The three seeds are independent random initialisations and land in different basins, so averaging
  their weights destroys the model. This is the expected outcome and is reported only to close the
  idea off.

## 6. Critical analysis

**What this establishes.** At 1M parameters and a 9.8–39.3M target budget, replacing a learned
absolute position table with rotary embeddings improves test BPB by 0.056 at matched processed
targets, with an effect several times the seed spread, at equal or lower parameter count, within all
three resource budgets. And the size of that advantage is **regime-dependent**: it is four times
larger at 1,200 steps than at 4,800.

**What it does not establish.**

- **Rotation vs relative structure.** RoPE is one way to obtain relative offsets. The `none` arm
  controls for parameter count but nothing here separates "relative offsets help" from "this
  particular rotation helps". A fixed additive relative bias (ALiBi) or a learned relative bias are
  the natural next arms; I would expect a fixed bias to capture much of the gain. **This is the single
  most informative experiment this study omits.**
- **Scale and setting.** One architecture family (width 128, depth 4), one context length (256), one
  dataset. The compute-limited regime is the stated reason the result should transfer, but that is an
  argument, not a measurement.
- **Three seeds.** The effect is several times the observed seed spread, but three seeds make the
  ±0.003–0.007 figures a rough scale, not a precise estimate. At 4,800 steps the `rope` advantage
  (0.0456) is six times that spread — clearly above noise, but no longer overwhelmingly so.
- **No hyperparameter search.** `base = 10000` and the half-split frequency layout are standard and
  were never tuned; the result is one design decision, not the outcome of a search.
- **Confounded capacity in two of three comparisons.** Only `rope` vs `none` holds parameters fixed.

**The result most likely to be over-read.** Table 1 and §4.2 show the model is still improving when
training stops, at both lengths. That means the 1.2M-parameter numbers quoted throughout are a
snapshot of an under-trained model, and the *relative* ordering of methods at this snapshot need not
survive longer training — the shrinking gap in §4.2 is direct evidence that it does not entirely.
Anyone extrapolating these comparisons to a converged model would be making a mistake this study
already demonstrates.

**Where the remaining gains are.** Given the compute-limited regime, I would look next at
*optimisation efficiency* rather than architecture: the supplied recipe uses AdamW at lr 1e-3 with
100 warmup steps into cosine decay, no weight-decay exemption for norms or biases, and no weight
averaging. EMA over the last part of a single trajectory (unlike the cross-seed average in §5.1) is
cheap, usually helps, costs nothing at inference, and composes with the change here. I did not test
it — a deliberate choice to keep the submitted model a single-variable result inside the time
available.

**On seed selection.** The submitted checkpoint is **seed 17**, `train.py`'s default and the seed used
throughout the supplied README examples — fixed before the sweep, not chosen from it. It happens to
be the best of the three seeds at both training lengths, which is disclosed rather than hidden; seed
19 scored 1.7027 and 1.9179 and was not used.

**On test-set usage.** All development and selection was on validation. The test split was scored
four times, each after the corresponding method was already frozen: two models at 1,200 steps and two
at 4,800 (method and baseline at each length). No test result changed any configuration,
hyperparameter or checkpoint choice — every model was trained and selected on validation before its
test score existed. A further scoring pass reproduced the submitted number from a fresh clone. Re-
scoring the same frozen predictor is what §7 invites.

## 7. Reproduction

```bash
cd MP1_student_starter/code
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v                # contract gate
```

```bash
# Reproduce the reported score from the submitted checkpoint, no retraining
python evaluate.py --checkpoint ../checkpoints/rope-s17-4800/checkpoint.pt \
  --device cpu --precision fp32 --threads 8 --split test

# The two comparison arms
python evaluate.py --checkpoint ../checkpoints/baseline-s17-4800/checkpoint.pt \
  --device cpu --precision fp32 --threads 8 --split test    # 1.7788
python evaluate.py --checkpoint ../checkpoints/baseline-s17-1200/checkpoint.pt \
  --device cpu --precision fp32 --threads 8 --split test    # 2.1027

# Budget measurements
python ../measure_peak_ram.py runs/long-rope-s17/checkpoint.pt validation
python ../make_run_log.py                                 # regenerate RUN_LOG.csv
bash ../run_matrix.sh                                     # retrain the 1,200-step matrix
```

The evaluator writes `test_cpu_fp32.json` beside the checkpoint; the **`bpb` field** in that file is
the submitted score. Timing figures in §5 use `--threads 8` and are thread-dependent; scores are not.

**Layout of the method.** `code/rope.py` (the rotation) and `code/student.py` (the model, with
`config['pos_mode']` selecting the arm) are the only files differing from the baseline;
`code/model.py`, `code/common.py`, `code/evaluate.py`, `code/tests/` and `code/data/` are untouched.
`code/configs/pos-*.json` mirror `baseline.json` plus one key, which `model.py` ignores.
`run_matrix.sh` covers the 1,200-step sweep; the 4,800-step runs used the same commands with
`--steps 4800` and `long-` prefixed run directories (see `RUN_LOG.csv`).

### On training reproducibility

I first reported that training here is bit-reproducible. That claim was too strong and is corrected
here. Four re-runs of `rope`/seed 17 at 4,800 steps found:

- **Three runs with identical arguments produced byte-identical checkpoints** (`128b8c0d…`), including
  runs whose `--eval-every` differed (1,200 vs 600), so the evaluation cadence is not what matters.
- **One run differs**, at every one of the 52 tensors, by at most ~7e-4, moving validation BPB by
  1.8e-5. That run was the only one trained while a second process was doing heavy CPU work
  concurrently. The plausible mechanism is that the changed memory-allocation conditions altered
  SDPA's kernel selection, and the three available backends do not sum in the same order.

So the accurate statement is: **training is deterministic when the machine is left alone, and stable
to ~2e-5 BPB when it is not** — two orders of magnitude below the seed spread, and irrelevant to any
conclusion here. It is not a guarantee, and the earlier wording should not be relied on.

**Reusing a checkpoint does not erase its training cost.** All figures are for models trained from
random initialisation for the full 9,830,400 or 39,321,600 targets; nothing is fine-tuned from an
ancestor.

### Acknowledgement of AI assistance

Claude (Anthropic) was used as a coding and analysis assistant for this coursework: drafting
`rope.py` and the `pos_mode` switch in `student.py`; writing the experiment driver, the peak-RAM
probe, the log generator and the ensemble probe; and drafting this report. All experimental design
decisions — the choice of RoPE over alternatives, the three-arm ablation with a parameter-matched
lower-bound control, running the sweep at two training lengths, the seed-selection policy, the
rejection of the ensemble on budget grounds, and the interpretation in §4 and §6 — were reviewed and
are the author's responsibility. Every number reported here was produced by the supplied harness on
the hardware in §2 and can be regenerated with the commands in §7. The author has verified that the
implementation is causal and contract-test compliant, that the `learned` arm is bitwise identical to
the supplied baseline, and that no test-set information influenced any development decision.

### References

- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, arXiv:2104.09864.
- Merity et al., *Pointer Sentinel Mixture Models*, arXiv:1609.07843 (WikiText-2).
- Loshchilov & Hutter, *Decoupled Weight Decay Regularization*, arXiv:1711.05101 (AdamW).
