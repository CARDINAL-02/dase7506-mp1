# DASE7506 MP1 — submission

**Frozen test BPB: 1.7228** (39,321,600 processed targets)

| arm | targets | test BPB |
|---|---:|---:|
| initial baseline (`model.py`, 1,200 steps) | 9,830,400 | 2.1027 |
| matched-target baseline (`model.py`, 4,800 steps) | 39,321,600 | 1.7788 |
| **submitted: RoPE, 4,800 steps** | 39,321,600 | **1.7228** |

Method: replace the baseline's learned absolute position embedding with rotary position embeddings
(RoPE), applied to queries and keys. This removes a 256×128 table (32,768 parameters, 3.0% of the
model) and builds *relative* position structure into attention instead of asking the optimiser to
discover it. At matched processed targets it is worth **−0.056 BPB**.

The supplied recipe's 1,200-step default is far below where the loss curve flattens, so the submitted
model also trains 4× longer — that accounts for the rest of the movement against the initial
baseline. It is a cost decision, not an algorithmic contribution, and the report says so.

The full write-up — including the ablation across two training lengths, the budget analysis, two
measured dead ends, and the limitations — is in [REPORT.md](REPORT.md). Per-run accounting for all 24
training runs is in [RUN_LOG.csv](RUN_LOG.csv).

## Layout

| Path | |
|---|---|
| `REPORT.md` | the report (≤10 pages) |
| `RUN_LOG.csv` | one row per experiment, all 24 runs |
| `checkpoints/` | `rope-s17-4800` (submitted), `baseline-s17-4800`, `baseline-s17-1200` |
| `run_matrix.sh` | reproduces the ten 1,200-step runs |
| `make_run_log.py` | regenerates `RUN_LOG.csv` from the run artefacts |
| `measure_peak_ram.py` | CPU peak-RAM probe (not instrumented by the harness) |
| `explore_ensemble.py` | the ensemble probe described in §5.1 of the report |
| `CODE_WALKTHROUGH.md` | line-by-line explanation of `rope.py` and `student.py`, with 10 self-test questions |
| `trace_rope.py` | regenerates every number quoted in the walkthrough |
| `code/` | the code package |
| `GUIDE.md` | the course assignment guide, as supplied |

Inside `code/`, **only two files differ from the supplied package**: `rope.py` (new) and
`student.py` (the model, with `config['pos_mode']` selecting the ablation arm). `model.py`,
`common.py`, `evaluate.py`, `tests/` and `data/` are untouched.

`code/configs/pos-{learned,none,rope}.json` mirror `configs/baseline.json` plus one `pos_mode` key.
`model.py` ignores that key, so either implementation can read either config.

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
CPU**. Nothing here needs network access beyond installing the dependencies.

## Reproduce the reported score (no retraining)

```bash
cd code
python evaluate.py --checkpoint ../checkpoints/rope-s17-4800/checkpoint.pt \
  --device cpu --precision fp32 --threads 8 --split test
```

This writes `test_cpu_fp32.json` beside the checkpoint. The **`bpb` field** in that file is the
submitted score, **1.7228**. The checkpoint's SHA256 is recorded in `RUN_LOG.csv` and echoed into the
JSON, so the number is tied to a specific artefact.

The same command on the two comparison arms:

```bash
python evaluate.py --checkpoint ../checkpoints/baseline-s17-4800/checkpoint.pt \
  --device cpu --precision fp32 --threads 8 --split test    # 1.7788, matched targets
python evaluate.py --checkpoint ../checkpoints/baseline-s17-1200/checkpoint.pt \
  --device cpu --precision fp32 --threads 8 --split test    # 2.1027, initial baseline
```

## Retrain

```bash
bash run_matrix.sh          # the ten 1,200-step runs
python make_run_log.py      # regenerate RUN_LOG.csv
```

The 4,800-step runs used the same commands with `--steps 4800` and `long-` prefixed run directories;
`RUN_LOG.csv` records every one. Each run is trained from random initialisation — 1,200 or 4,800
steps × 32 sequences × 256 targets. `train.py` refuses to write into a non-empty `--run-dir`, so every
run needs a fresh directory. Total GPU training across all 24 logged runs is 27.2 minutes.

## Reproducibility notes

- `code/data/` is SHA256-verified against `data/manifest.json` on every load. The repository root
  carries a `.gitattributes` with `* -text`; on Windows `core.autocrlf=true` would otherwise rewrite
  line endings on checkout, change the bytes and make the evaluator reject the benchmark.
- The contract tests build the model at `width=32, heads=4, depth=2` and run it on length-12
  sequences, so any position mechanism must work far from the submitted configuration. `rope.py`
  derives its frequency schedule from `head_dim` at runtime for this reason, and its tables are
  non-persistent buffers so they stay out of the checkpoint.
- The `learned` arm is bitwise identical to `model.py`'s GPT — verified by identical `state_dict`
  keys, identical outputs after loading the baseline's weights, and an identical validation BPB at
  both training lengths (2.0729 and 1.7532 for `learned`-s17 and the supplied baseline alike). The
  ablation therefore changes only how position reaches attention.
- Training is deterministic when the machine is left alone: three runs of `rope`/seed 17 at 4,800
  steps with identical arguments produced byte-identical checkpoints. One further run, trained while
  a second process was busy on the same machine, deviated by ~7e-4 per weight — worth 1.8e-5 BPB, two
  orders of magnitude below the seed spread. The report's §7 documents this; the earlier claim of
  unconditional bit-reproducibility was too strong and has been withdrawn.

## Reused work

- **`code/student.py` is a modified copy of the supplied `code/model.py`.** The `Block` and `GPT`
  classes are the course baseline's, carried over unchanged; the additions are a `pos_mode` switch
  and a conditional position embedding. The `learned` arm is bit-identical to the original — see
  §2.1 of the report.
- **The rotary position embedding method is not mine.** It is Su et al., *RoFormer: Enhanced
  Transformer with Rotary Position Embedding*, arXiv:2104.09864.
- **`code/common.py`, `code/evaluate.py`, `code/train.py`, `code/tests/` and `code/data/` are the
  course's**, used unmodified. `rope.py`, `run_matrix.sh`, `make_run_log.py`, `measure_peak_ram.py`
  and `explore_ensemble.py` are new work for this submission.
- **WikiText-2** is Merity et al., arXiv:1609.07843; data and tokenizer are redistributed unmodified
  as supplied. See the data attribution below.

## Acknowledgement of AI assistance

Claude (Anthropic) was used as a coding and analysis assistant: drafting `code/rope.py` and the
`pos_mode` switch in `code/student.py`; writing `run_matrix.sh`, `measure_peak_ram.py`,
`make_run_log.py` and `explore_ensemble.py`; and drafting `REPORT.md`. The experimental design — the
choice of RoPE, the three-arm ablation with a parameter-matched lower-bound control, running the
sweep at two training lengths, the seed-selection policy, the rejection of the ensemble on budget
grounds, and the interpretation — was reviewed and remains the author's responsibility. Every
reported number was produced by the supplied harness and can be regenerated with the commands above;
no test-set information influenced any development decision. See §7 of the report for the full
statement.

## Data attribution

WikiText-2 (Merity et al., arXiv:1609.07843), CC BY-SA 3.0 / GNU FDL. Data and tokenizer are
redistributed here unmodified as supplied by the course; `data/manifest.json` records their hashes.
