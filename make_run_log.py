"""Build RUN_LOG.csv from the run artefacts.

Every cell is read back out of metrics.json / the evaluator JSONs rather than
typed by hand, so the log cannot drift from what actually ran.

    python make_run_log.py
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / 'code' / 'runs'
CODE_COMMIT = 'a5e9a86'          # the commit holding the model code for every run
HARDWARE = 'RTX 3060 Laptop 6GB'
THREADS = 4                      # train.py default; a separate CPU pass measures the 5x budget

COLUMNS = ['run_id', 'code_commit', 'method', 'config_path', 'seed', 'parent_checkpoints',
           'steps', 'batch_size', 'context', 'processed_targets_including_ancestry',
           'parameters', 'hardware', 'threads', 'train_precision', 'train_seconds',
           'validation_seconds', 'test_seconds', 'peak_ram_gb', 'peak_gpu_allocated_gb',
           'validation_bpb', 'test_bpb', 'checkpoint_sha256', 'selected_on', 'notes']

# The three timing columns and peak RAM are stated on CPU FP32, which is the
# device the evaluation budgets are defined on. Training happened on CUDA, so the
# ~0.5 s validation passes that ran during training are NOT comparable and are
# recorded in notes instead of silently sharing a column with CPU numbers.
CPU_PASS = {   # validation_s, test_s, peak_RAM_GiB -- measure_peak_ram.py / evaluate.py
    'baseline-model-s17': (9.21, 10.51, 1.784),
    'rope-s17': (10.77, 11.89, 1.785),
}


def row_for(run_dir):
    m = json.loads((run_dir / 'metrics.json').read_text())
    is_student = m['implementation'] == 'student'
    mode = m['config'].get('pos_mode', 'rope') if is_student else 'baseline'
    test = run_dir / 'test_cpu_fp32.json'
    test_bpb = json.loads(test.read_text())['bpb'] if test.exists() else ''
    steps = m['train_tokens'] // (32 * 256)
    cpu_val_s, cpu_test_s, cpu_ram = CPU_PASS.get(run_dir.name, ('not run on CPU', '', ''))
    return {
        'run_id': run_dir.name,
        'code_commit': CODE_COMMIT,
        'method': 'supplied baseline (model.py)' if not is_student else f'student, pos_mode={mode}',
        'config_path': 'code/configs/baseline.json' if not is_student else f'code/configs/pos-{mode}.json',
        'seed': m['seed'],
        'parent_checkpoints': 'none (from random init)',
        'steps': steps,
        'batch_size': 32,
        'context': m['config']['context'],
        'processed_targets_including_ancestry': m['train_tokens'],
        'parameters': m['parameters'],
        'hardware': HARDWARE,
        'threads': THREADS,
        'train_precision': m['precision'],
        'train_seconds': round(m['train_seconds'], 1),
        'validation_seconds': cpu_val_s,
        'test_seconds': cpu_test_s,
        'peak_ram_gb': cpu_ram,
        'peak_gpu_allocated_gb': round(m.get('peak_allocated_gb', 0.0), 3),
        'validation_bpb': round(m['validation']['bpb'], 4),
        'test_bpb': round(test_bpb, 4) if test_bpb != '' else '',
        'checkpoint_sha256': m['checkpoint_sha256'],
        'selected_on': 'validation',
        'notes': '; '.join(filter(None, [
            ('frozen submission; test scored once' if run_dir.name == 'rope-s17' else
             'test scored for the baseline comparison' if run_dir.name == 'baseline-model-s17' else
             'ablation sweep, validation only'),
            ('' if run_dir.name in CPU_PASS else
             f'CUDA validation pass during training: {m["validation"]["seconds"]:.2f} s '
             f'(not the budget metric)'),
            (f'CPU pass is {cpu_val_s / CPU_PASS["baseline-model-s17"][0]:.2f}x the baseline, budget 5x'
             if run_dir.name == 'rope-s17' else ''),
        ])),
    }


def main():
    rows = [row_for(d) for d in sorted(RUNS.iterdir())
            if (d / 'metrics.json').exists() and not d.name.startswith('smoke')]
    out = ROOT / 'RUN_LOG.csv'
    with out.open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {out} ({len(rows)} runs)')


if __name__ == '__main__':
    main()
