# DASE7506 MP1 — submission

**Frozen test BPB: 1.9527** · supplied baseline: 2.1027 · **−0.1500 BPB (−7.13%)**

Method: replace the baseline's learned absolute position embedding with rotary position embeddings
(RoPE), applied to queries and keys. This removes a 256×128 table (32,768 parameters, 3.0% of the
model) and builds *relative* position structure into attention instead of asking the optimiser to
discover it.

The full write-up, including the ablation, the cost analysis and the limitations, is in
[REPORT.md](REPORT.md). Per-run accounting is in [RUN_LOG.csv](RUN_LOG.csv).

## Layout

| Path | |
|---|---|
| `REPORT.md` | the report (≤10 pages) |
| `RUN_LOG.csv` | one row per experiment |
| `run_matrix.sh` | reproduces all ten training runs |
| `make_run_log.py` | regenerates `RUN_LOG.csv` from the run artefacts |
| `measure_peak_ram.py` | CPU peak-RAM probe (not instrumented by the harness) |
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
python evaluate.py --checkpoint ../checkpoints/rope-s17/checkpoint.pt \
  --device cpu --precision fp32 --threads 8 --split test
```

This writes `test_cpu_fp32.json` beside the checkpoint. The **`bpb` field** in that file is the
submitted score, **1.9527**. The checkpoint's SHA256 is recorded in `RUN_LOG.csv` and echoed into
the JSON, so the number is tied to a specific artefact.

For the baseline arm of the comparison, the same command on
`../checkpoints/baseline-model-s17/checkpoint.pt` gives **2.1027**.

## Retrain everything

```bash
bash run_matrix.sh          # ten runs, ~4.25 min GPU training, ~7 min wall clock on an RTX 3060
python make_run_log.py      # regenerate RUN_LOG.csv
```

Each run is 1,200 steps × 32 sequences × 256 targets = 9,830,400 processed targets from random
initialisation. `train.py` refuses to write into a non-empty `--run-dir`, so every run needs a fresh
directory.

## Reproducibility notes

- `code/data/` is SHA256-verified against `data/manifest.json` on every load. The repository root
  carries a `.gitattributes` with `* -text`; on Windows `core.autocrlf=true` would otherwise rewrite
  line endings on checkout, change the bytes and make the evaluator reject the benchmark.
- The contract tests build the model at `width=32, heads=4, depth=2` and run it on length-12
  sequences, so any position mechanism must work far from the submitted configuration. `rope.py`
  derives its frequency schedule from `head_dim` at runtime for this reason, and its tables are
  non-persistent buffers so they stay out of the checkpoint.
- The `learned` arm is bitwise identical to `model.py`'s GPT — verified by identical `state_dict`
  keys, identical outputs after loading the baseline's weights, and an identical validation BPB
  (2.0729 for both `learned`-s17 and the supplied baseline). The ablation therefore changes only how
  position reaches attention.

## Acknowledgement of AI assistance

Claude (Anthropic) was used as a coding and analysis assistant: drafting `code/rope.py` and the
`pos_mode` switch in `code/student.py`; writing `run_matrix.sh`, `measure_peak_ram.py` and
`make_run_log.py`; and drafting `REPORT.md`. The experimental design — the choice of RoPE, the
three-arm ablation with a parameter-matched lower-bound control, the seed-selection policy and the
interpretation — was reviewed and remains the author's responsibility. Every reported number was
produced by the supplied harness and can be regenerated with the commands above; no test-set
information influenced any development decision. See §7 of the report for the full statement.

## Data attribution

WikiText-2 (Merity et al., arXiv:1609.07843), CC BY-SA 3.0 / GNU FDL. Data and tokenizer are
redistributed here unmodified as supplied by the course; `data/manifest.json` records their hashes.
