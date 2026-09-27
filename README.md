# DASE7506 MP1 — submission

**Frozen test BPB: 1.5544** (314,572,800 processed targets)

| arm | targets | test BPB |
|---|---:|---:|
| initial baseline (`model.py`, 1,200 steps) | 9,830,400 | 2.1027 |
| v1: RoPE position encoding, 1.05M params, 4,800 steps | 39,321,600 | 1.7228 |
| **v2: RoPE + dropout + 2.8M params, 38,400 steps** | 314,572,800 | **1.5544** |

## What this submission claims

The stated evaluation budgets cover *inference* only — CPU scoring ≤ 5× baseline, RAM ≤ 4 GiB,
assets ≤ 64 MiB — and training is unrestricted. It is tempting to read that as "scale up until the
inference budget binds". **The measurements say that is the wrong lever.**

The corpus is **3,613,343 training tokens**. The supplied 1,200-step recipe is already 10.9 epochs
of it. Growing the model past ~2M parameters stops helping, and at 2.8M parameters doubling the step
count makes validation BPB *worse* (1.6509 → 1.6751). Train/validation loss gaps grow with both
size and steps. The binding constraint is **generalisation on a small corpus**, not capacity and not
the asset budget — v1 used 4.3% of the 64 MiB allowance.

Two changes follow from that diagnosis:

1. **Dropout** (embedding output and both residual branches). This alone is worth **−0.103 BPB** at
   fixed shape and step count — an order of magnitude more than the position-encoding change at this
   scale. It is what makes capacity usable at all.
2. **Scale, spent where the data allows it**: width 160 / depth 8 / 8 heads (2,802,240 parameters),
   trained 38,400 steps.

The position encoding is unchanged from v1: **rotary position embeddings (RoPE)** replacing the
baseline's learned absolute position table. The ablation is re-established at the v2 configuration
in the report.

The full write-up — diagnosis, ablation, cost analysis, two measured dead ends, and the
limitations — is in [REPORT.md](REPORT.md). Per-run accounting is in [RUN_LOG.csv](RUN_LOG.csv).

## Layout

| Path | |
|---|---|
| `REPORT.md` | the report (≤10 pages) |
| `RUN_LOG.csv` | one row per experiment |
| `checkpoints/` | the submitted model plus every comparison arm |
| `run_matrix.sh` | reproduces the v1 1,200-step sweep |
| `make_run_log.py` | regenerates `RUN_LOG.csv` from the run artefacts |
| `measure_peak_ram.py` | CPU peak-RAM probe (not instrumented by the harness) |
| `explore_ensemble.py` | the ensemble probe described in §5.1 of the report |
| `code/` | the code package |
| `GUIDE.md` | the course assignment guide, as supplied |

Inside `code/`, the files differing from the supplied package are `rope.py` (new), `student.py` (the
model, with `config['pos_mode']` and `config['dropout']`) and `train.py` (four new optimisation
flags). `model.py`, `common.py`, `evaluate.py`, `tests/` and `data/` are untouched.

`code/configs/pos-*.json` and `code/configs/v2-*.json` mirror `baseline.json` plus the keys each arm
needs; `model.py` ignores keys it does not know.

## Install

Python 3.12. From this directory:

```bash
cd code
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Substitute the `cu126` index for GPU training. Training may use BF16; **ranked evaluation is FP32 on
CPU**.

## Reproduce the reported score (no retraining)

```bash
cd code
python evaluate.py --checkpoint ../checkpoints/v2-submission/checkpoint.pt \
  --device cpu --precision fp32 --threads 8 --split test
```

This writes `test_cpu_fp32.json` beside the checkpoint. The **`bpb` field** in that file is the
submitted score. The checkpoint's SHA256 is recorded in `RUN_LOG.csv` and echoed into the JSON, so
the number is tied to a specific artefact.

## Retrain

```bash
bash run_matrix.sh          # the v1 1,200-step sweep
python make_run_log.py      # regenerate RUN_LOG.csv
```

The v2 runs used the same command with `--config configs/v2-w160h5d8-do10.json --steps 38400`.
`RUN_LOG.csv` records every run. `train.py` refuses to write into a non-empty `--run-dir`, so each
run needs a fresh directory.

## Reproducibility notes

- `code/data/` is SHA256-verified against `data/manifest.json` on every load. The repository root
  carries a `.gitattributes` with `* -text`; on Windows `core.autocrlf=true` would otherwise rewrite
  line endings on checkout, change the bytes and make the evaluator reject the benchmark.
- The contract tests build the model at `width=32, heads=4, depth=2` on length-12 sequences, so any
  position mechanism must work far from the submitted configuration. `rope.py` derives its frequency
  schedule from `head_dim` at runtime for this reason, and its tables are non-persistent buffers.
- Every new option defaults to the original recipe. With `dropout` unset and no new flags, a full
  4,800-step run reaches a state dict **identical tensor for tensor** to the frozen v1 checkpoint
  (0 of 52 tensors differ). Verified before use.
- Adding a dropout module changes the checkpoint's **bytes** even at `p = 0`, because `torch.save`
  pickles the module tree. The weights are unaffected. Byte-identity therefore holds per code
  revision, not across revisions that add modules.

## Reused work

- **`code/student.py` is a modified copy of the supplied `code/model.py`.** The `Block` and `GPT`
  classes are the course baseline's; the additions are a `pos_mode` switch, a `dropout` scalar and a
  `make_dropout` helper. The `learned` arm is bit-identical to the original.
- **The rotary position embedding method is not mine.** It is Su et al., *RoFormer: Enhanced
  Transformer with Rotary Position Embedding*, arXiv:2104.09864.
- **`code/common.py`, `code/evaluate.py`, `code/tests/` and `code/data/` are the course's**, used
  unmodified. `rope.py`, `run_matrix.sh`, `make_run_log.py`, `measure_peak_ram.py` and
  `explore_ensemble.py` are new work for this submission.
- **WikiText-2** is Merity et al., arXiv:1609.07843; data and tokenizer are redistributed unmodified
  as supplied.

## Acknowledgement of AI assistance

Claude (Anthropic) was used as a coding and analysis assistant: it wrote `code/rope.py`, the model
and training changes and the experiment scripts, and drafted `REPORT.md`. The author set the
objectives and the direction of the work, reviewed the changes, and is responsible for what is
submitted. Every reported number was produced by the supplied harness and can be regenerated with
the commands above.

## Data attribution

WikiText-2 (Merity et al., arXiv:1609.07843), CC BY-SA 3.0 / GNU FDL. Data and tokenizer are
redistributed here unmodified as supplied by the course; `data/manifest.json` records their hashes.
