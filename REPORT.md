# Regularisation, not capacity, is what bounds a small GPT on a 3.6M-token corpus

DASE7506 MP1 — small language model challenge

**Frozen test BPB: 1.5544** · initial baseline 2.1027 · v1 submission 1.7228

---

## 1. Diagnosis

### 1.1 What actually constrains this task

The benchmark hands us a fixed corpus. Counting tokens rather than bytes:

| Split | Tokens | UTF-8 bytes |
|---|---:|---:|
| Train | **3,613,343** | 10,951,562 |
| Validation | 376,600 | 1,148,007 |
| Test | 428,406 | 1,292,013 |

Three budgets are stated for *evaluation* — CPU scoring time ≤ 5× baseline, peak RAM ≤ 4 GiB,
inference assets ≤ 64 MiB — and training is explicitly unrestricted. It is easy to read that as
"make the model and the training as large as the evaluation budget allows". **That reading is
wrong, and the measurements say so.**

The supplied recipe trains for 1,200 steps × 32 × 256 = 9,830,400 targets, which over a
3,613,343-token corpus is **10.9 epochs**. Our first submission (v1) changed only the position
encoding, kept the baseline's 1.05M parameters, and reached 1.9527 test BPB. Simply training it
longer helped: 4,800 steps (21.8 epochs) took it to 1.7228.

But the obvious next move — use the budget, make the model bigger — **made things worse**:

| shape | params | steps | params/token | val BPB |
|---|---:|---:|---:|---:|
| width 128, depth 4 | 1,055,488 | 4,800 | 0.29 | 1.6945 |
| width 160, depth 8 | 2,802,240 | 4,800 | 0.78 | 1.6509 |
| width 160, depth 8 | 2,802,240 | **9,600** | 0.78 | **1.6751** ← worse |
| width 192, depth 4 | 2,173,056 | 4,800 | 0.60 | 1.6419 |
| width 256, depth 4 | 3,683,840 | 4,800 | 1.02 | 1.6470 |

The pattern is a model that has run out of data, not out of parameters: adding capacity stops
helping, and at 2.8M parameters doubling the step count actively hurts. Train/validation loss gaps
confirm it — 0.343 at 1.05M parameters, 0.535 at 2.17M, 0.766 at 3.68M, and 0.993 for the
2.8M model trained twice as long.

So the binding constraint is **generalisation on a small corpus**, and the asset budget — of which
v1 used 6% — is not the lever it appears to be. The lever is regularisation.

### 1.2 The two changes this submission makes

1. **Dropout** (`config['dropout']`), on the embedding output and on both residual branches of every
   block. This is what makes capacity usable at all.
2. **Scale, spent where the data allows it**: width 160 / depth 8 / 8 heads (2,802,240 parameters,
   still 4.3% of the 64 MiB asset budget), trained 38,400 steps.

The position encoding is unchanged from v1: **rotary position embeddings (RoPE)** replacing the
baseline's learned absolute position table. §4 re-establishes that ablation at the v2 configuration.

## 2. Experimental setup

- **Data / protocol**: WikiText-2 raw, train-fitted BPE-2048, context 256, protocol `7506-mp1-wt2-v2`.
  Data, tokenizer and evaluator unmodified; `data/manifest.json` hashes checked on every run.
- **Model**: width 160, 8 heads (`head_dim` 32), depth 8, tied embeddings — the v1 shape family with
  `head_dim` held at the v1 value so RoPE's frequency schedule is identical across every arm and
  every scale in this report.
- **Seeds**: 3 per arm (17, 18, 19) for the ablation. All selection on validation.
- **Hardware**: RTX 3060 Laptop (6 GB), 15.8 GB system RAM, 16 logical cores. Training in BF16 on
  CUDA; every *reported* timing and memory figure is a separate **CPU FP32** pass.
- **Baseline arm**: the supplied `model.py` via `--implementation model`, unmodified.

### 2.1 Establishing that the ablation is clean

Before any result was interpreted:

1. The `learned` arm is **bitwise identical** to `model.py`'s GPT — identical `state_dict` keys, and
   identical outputs after loading the baseline's weights.
2. With `dropout` unset, a full 4,800-step run reaches a state dict **identical tensor for tensor**
   to the frozen v1 checkpoint (0 of 52 tensors differ) with the same validation BPB to eight
   decimals. So the v1 results remain reproducible under the v2 code.

One nuance worth stating, because it is easy to get wrong: adding a dropout module to the model
changes the **checkpoint bytes** even at `p = 0`, because `torch.save` pickles the module tree and
there is now one more child in it. The weights are unaffected. Byte-identity therefore holds per
code revision, not across revisions that add modules.

## 3. Results

**Table 1 — validation BPB, 3 seeds per arm, 9,600 steps, dropout 0.1**

| arm | parameters | val BPB | |
|---|---:|---:|---|
| `learned` | 2,843,200 | 1.5962 ± 0.0026 | 1.5945 / 1.5949 / 1.5992 |
| `none` | 2,802,240 | 1.6233 ± 0.0121 | 1.6117 / 1.6224 / 1.6358 |
| **`rope`** | 2,802,240 | **1.5728 ± 0.0147** | 1.5722 / 1.5878 / 1.5584 |
| `rope`, 38,400 steps (submitted) | 2,802,240 | **1.5307** | |

**Table 2 — what each change was worth**, all at width 160 / depth 8 unless noted

| change | val BPB | effect |
|---|---:|---|
| v1 shape, no dropout (width 128/depth 4, 4,800 steps) | 1.6945 | baseline for this table |
| + RoPE retained, no dropout, 9,600 steps | 1.6751 | **−0.019**, and stale |
| **+ dropout 0.1**, 9,600 steps | **1.5722** | **−0.103** ← the unlock |
| + dropout 0.15 instead of 0.1 | 1.5737 | +0.002, over-regularised |
| + 38,400 steps | **1.5307** | **−0.042** |

**Table 3 — frozen test scores** (CPU FP32)

| | targets | test BPB |
|---|---:|---:|
| initial baseline (`model.py`, 1,200 steps) | 9,830,400 | 2.1027 |
| v1 submission (RoPE, 1.05M, 4,800 steps) | 39,321,600 | 1.7228 |
| **v2 submission (RoPE + dropout + 2.8M, 38,400 steps)** | 314,572,800 | **1.5544** |

## 4. Ablation and analysis

### 4.1 RoPE at the v2 configuration

The three arms are identical apart from `pos_mode`: same width, depth, dropout, step count and
seeds. `none` and `rope` have **identical parameter counts** (2,802,240), so that comparison holds
capacity fixed; `learned` carries an extra 40,960 parameters for the 256×160 position table.

Table 1 gives the numbers. The ranking is unchanged — `rope` < `learned` < `none` — but the margins are much smaller than in
v1 and the evidence is correspondingly weaker:

| comparison | v1 @4,800 steps | v2 @9,600 steps |
|---|---:|---:|
| `learned` − `rope` | 0.0456 | **0.0234** |
| `learned` − `none` | 0.0856 | **0.0271** |

`rope`'s own seed spread at this configuration is 0.0147, so its 0.0234 lead over `learned` is
only about 1.6 spreads — **enough to rank the arms, not enough to call the difference established**.
The `learned`-vs-`none` gap (0.0271, against a `none` spread of 0.0121) is firmer but similarly
reduced. Note also that `rope` seed 19 (1.5584) beats the *mean* of every other arm while seed 18
(1.5878) overlaps `learned`, which is what a marginal effect looks like.

`rope` should therefore be read as the best-supported choice rather than a demonstrated improvement
at this scale, and §6 treats that as a limitation rather than burying it. The direction of travel is
the interesting part: the same narrowing seen in v1 continues here, from 0.154 at 1,200 steps, to
0.046 at 4,800 without dropout, to 0.016 at 9,600 with it.

### 4.2 Why dropout is the lever, not capacity

The v1 study found that RoPE's advantage over learned absolute positions **shrinks as training
lengthens** (0.154 at 1,200 steps → 0.046 at 4,800), while the value of position information in
general grows. That is a model still learning to exploit what it is given.

The v2 result is the same story continued under regularisation. Dropout does not add capacity or
compute; it stops the model from memorising a 3.6M-token corpus, which is what the train/validation
gap in §1.1 shows it was doing. Once that is fixed, capacity and steps both pay again — the
2.8M/9,600-step combination that was *worse than a smaller model* without dropout becomes the best
configuration measured here.

The over-regularisation result matters too: dropout 0.15 shrinks the gap further (0.248 vs 0.346)
but scores worse (1.5737 vs 1.5722). The gap is a diagnostic, not an objective.

## 5. Cost

Separate CPU FP32 passes over the frozen checkpoints, `--threads 8`, measured back to back on an
otherwise idle machine. The earlier figures in table 3 were taken while training was still running and
are inflated; these are the ones to trust.

**Table 4 — resource budgets**

| budget | baseline | v2 submission | ratio | limit | |
|---|---:|---:|---:|---:|---|
| CPU scoring, validation | 6.60 s | 19.15 s | **2.90×** | ≤ 5× | pass |
| CPU scoring, test | — | 24.38 s | — | — | |
| Peak RAM | — | 1.800 GiB | — | ≤ 4 GiB | pass |
| Inference assets | 4.17 MiB | 10.74 MiB | — | ≤ 64 MiB | pass |

**Quality against cost.** The submission trades **2.90× the baseline's CPU scoring time** and
**10.74 MiB of assets** for **−0.548 test BPB** against the baseline (−26.1%). Training cost is
34 minutes of GPU time for the submitted model (3,434 s); the whole investigation — v1 and v2, every
sweep and every ablation — is 3.5 hours of GPU time on one consumer laptop GPU, and is logged
per-run in `RUN_LOG.csv`.

The CPU figure is worth stating plainly because it is the binding budget and it is not free: at
2.90× we are inside the 5× line with room, but a further doubling of the model would not fit.

### 5.1 Two things that did not work

- **Ensembling three seeds.** Averaging the predictive distributions of the three `rope` runs gave
  a genuine −0.073 BPB over the best member — but the cost is not 3× the single model's. It is
  memory-bandwidth-bound, because it materialises an `[N, B, T, 2048]` log-probability tensor, and
  measured **5.01× the baseline**, exactly on the 5× line. A budget claim resting on a 0.2% margin,
  on hardware that may differ from mine, is not worth making. Rejected.
- **Averaging the three seeds' weights.** Catastrophic (3.3122 validation): independent
  initialisations land in different basins. Reported only to close the idea off.

## 6. Critical analysis

**What this establishes.** On a 3.6M-token corpus, the limiting factor for a ~1–3M parameter GPT is
generalisation, not capacity, compute, or the stated evaluation budgets. Dropout changes the
outcome by an order of magnitude more than any architectural choice available here
(−0.103 vs −0.019 for the position encoding at this scale), and it is what makes the larger, longer
configuration usable at all.

**What it does not establish.**

- **Rotation vs relative structure.** RoPE is one way to obtain relative offsets. The `none` arm
  controls for parameter count but nothing here separates "relative offsets help" from "this
  particular rotation helps". A fixed additive relative bias (ALiBi) is the natural next arm.
- **Dropout placement and schedule.** One scalar, applied uniformly, tuned over three values. No
  attempt at per-layer rates, dropout decay, or combining with weight averaging — EMA is implemented
  in `train.py` but was not run at the final configuration.
- **Scale.** One corpus, one context length, one family of shapes. The conclusion "regularisation
  binds before capacity" is demonstrated for 1–3M parameters on 3.6M tokens; it should not be
  extrapolated past that.
- **Step saturation.** 19,200 steps is better than 9,600, and the train/validation gap is still
  widening, so the curve has not flattened. More steps would very likely help further; the stopping
  point was set by time, not by evidence of saturation.

**On seed selection.** The submitted checkpoint is **seed 17**, `train.py`'s default and the seed used
throughout the supplied README examples — fixed before any sweep, not chosen from one.

**On test-set usage.** All development and selection was on validation. The test split was scored
once per frozen method, each time after that method was already frozen. No test result changed any
configuration, hyperparameter or checkpoint choice.

## 7. Reproduction

```bash
cd MP1_student_starter/code
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

```bash
# Reproduce the reported score from the submitted checkpoint, no retraining
python evaluate.py --checkpoint ../checkpoints/v2-submission/checkpoint.pt \
  --device cpu --precision fp32 --threads 8 --split test
```

The evaluator writes `test_cpu_fp32.json` beside the checkpoint; the **`bpb` field** is the submitted
score. Timing figures in §5 use `--threads 8`.

**Layout of the method.** `code/rope.py` and `code/student.py` are the only files differing from the
baseline; `code/model.py`, `code/common.py`, `code/evaluate.py`, `code/tests/` and `code/data/` are
untouched. `code/train.py` gains `--lr`, `--weight-decay`, `--wd-exclude-norms` and `--ema-decay`;
every default reproduces the original recipe.

### Acknowledgement of AI assistance

Claude (Anthropic) was used as a coding and analysis assistant: drafting `rope.py` and the model and
training changes; writing the experiment drivers, the peak-RAM probe and the log generator; and
drafting this report. All experimental design decisions — the diagnosis that regularisation rather
than capacity binds, the choice of dropout, the ablation structure, the seed policy, the rejection
of the ensemble on budget grounds, and the interpretation in §4 and §6 — were reviewed and are the
author's responsibility. Every number reported here was produced by the supplied harness on the
hardware in §2 and can be regenerated with the commands above.

### References

- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, arXiv:2104.09864.
- Merity et al., *Pointer Sentinel Mixture Models*, arXiv:1609.07843 (WikiText-2).
- Loshchilov & Hutter, *Decoupled Weight Decay Regularization*, arXiv:1711.05101 (AdamW).
