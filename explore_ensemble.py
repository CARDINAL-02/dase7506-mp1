"""Scratch probe: does averaging several rope checkpoints beat any single one?

Zero training cost -- the members already exist. Reports validation BPB and the
CPU FP32 scoring time, because an ensemble buys its gain with evaluation budget
and that trade is the whole question.

    python explore_ensemble.py rope-s17 rope-s18 rope-s19
    python explore_ensemble.py long-rope-s17 long-rope-s18 long-rope-s19
"""
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'code'))

from common import load_data, make_model, setup, windows   # noqa: E402


class Ensemble(torch.nn.Module):
    """Arithmetic mean of the member predictive distributions."""

    def __init__(self, members):
        super().__init__()
        self.members = torch.nn.ModuleList(members)
        self.context = members[0].context

    @torch.no_grad()
    def predict_log_probs(self, ids):
        stacked = torch.stack([m.predict_log_probs(ids).float() for m in self.members])
        return torch.logsumexp(stacked, dim=0) - math.log(len(self.members))


@torch.no_grad()
def bpb_of(model, tokens, byte_count, device):
    total, started = 0.0, time.perf_counter()
    for x, y in windows(tokens):
        logp = model.predict_log_probs(x.to(device)).float()
        losses = -logp.gather(-1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1).to(device)
        total += losses.masked_fill(y.to(device) == -100, 0).double().sum().item()
    return total / math.log(2) / byte_count, time.perf_counter() - started


def load(run, device):
    saved = torch.load(ROOT / 'code' / 'runs' / run / 'checkpoint.pt',
                       map_location='cpu', weights_only=True)
    model, _ = make_model(saved['implementation'], saved['config'], device)
    model.load_state_dict(saved['model'])
    return model.eval()


def main():
    runs = sys.argv[1:] or ['rope-s17', 'rope-s18', 'rope-s19']
    device, _ = setup('cpu', 'fp32', 8)
    tokens, byte_count = load_data()['validation']
    models = [load(r, device) for r in runs]
    base_bpb = 2.0729 if len(runs) == 3 and runs[0] == 'rope-s17' else None

    singles = []
    for name, model in zip(runs, models):
        value, seconds = bpb_of(model, tokens, byte_count, device)
        singles.append(value)
        print(f'  {name:16s} bpb={value:.4f}  cpu_score={seconds:6.2f}s')
    print(f'  {"best single":16s} bpb={min(singles):.4f}')

    for size in range(2, len(models) + 1):
        value, seconds = bpb_of(Ensemble(models[:size]), tokens, byte_count, device)
        print(f'  {"ensemble of " + str(size):16s} bpb={value:.4f}  cpu_score={seconds:6.2f}s')
    if base_bpb:
        print(f'  (1200-step baseline for scale: {base_bpb:.4f})')


if __name__ == '__main__':
    main()
