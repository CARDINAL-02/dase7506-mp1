"""Regenerate every number quoted in CODE_WALKTHROUGH.md.

The walkthrough asserts specific shapes, values and spreads. This script is how
those assertions were produced, so they can be re-checked rather than trusted.

    cd code && ../.venv/Scripts/python.exe ../trace_rope.py     # Windows venv
    cd code && python ../trace_rope.py                          # activated venv

Nothing here is part of the submission's inference path; it only reads the
model classes.
"""
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent / 'code'))

import student                                       # noqa: E402
from rope import apply_rope, build_rope              # noqa: E402

HEAD_DIM, CONTEXT = 32, 256


def section(title):
    print()
    print('=' * 72)
    print(title)
    print('=' * 72)


def trace_build_rope():
    section('TRACE 1  build_rope(head_dim=32, context=256)')
    idx = torch.arange(0, HEAD_DIM, 2).float()
    inv_freq = 1.0 / (10000.0 ** (idx / HEAD_DIM))
    freqs = torch.outer(torch.arange(CONTEXT).float(), inv_freq)
    print(f'  arange(0, head_dim, 2)          {tuple(idx.shape)}   {idx[:5].tolist()} ...')
    print(f'  inv_freq[0]  = {inv_freq[0]:.6f}          (fastest: 1 radian per token)')
    print(f'  inv_freq[-1] = {inv_freq[-1]:.8f}       (slowest)')
    print(f'  outer(arange(256), inv_freq)    {tuple(freqs.shape)}  [position, pair]')
    for t in (0, 1, 100):
        print(f'    freqs[{t}]'[:16] + f' = {[round(v, 4) for v in freqs[t][:4].tolist()]}')
    return freqs.cos(), freqs.sin()


def trace_relative_property():
    section('TRACE 2  the score depends only on the gap (the whole point)')
    # A dedicated head_dim=8 table. Frequencies are 10000^(-2j/head_dim), so they
    # depend on head_dim -- slicing the 32-dim table above would give different
    # numbers for the same positions.
    dim = 8
    c, s = build_rope(dim, CONTEXT)

    def rot(x, pos):
        cc, ss = c[pos], s[pos]
        e, o = x[..., 0::2], x[..., 1::2]
        return torch.stack((e * cc - o * ss, e * ss + o * cc), -1).flatten(-2)

    torch.manual_seed(0)
    q, k = torch.randn(dim), torch.randn(dim)
    print(f'  {"(m,n)":>10s} {"gap":>5s} {"score":>12s}')
    for m, n in [(0, 0), (1, 1), (5, 5), (0, 1), (4, 5), (10, 11), (0, 7), (20, 27)]:
        print(f'  {str((m, n)):>10s} {n - m:>5d} {(rot(q, m) * rot(k, n)).sum().item():>12.6f}')
    print('  lengths preserved:')
    print('    in : ' + '  '.join(f'{q.norm():.6f}' for _ in range(1)))
    print('    out: ' + '  '.join(f'{rot(q, m).norm():.6f}' for m in (0, 5, 17)))


def trace_apply_rope():
    section('TRACE 2b  apply_rope on [1,1,6,32]: position 0 is left untouched')
    torch.manual_seed(0)                 # build_rope consumes no RNG, so this is enough
    cos, sin = build_rope(HEAD_DIM, CONTEXT)
    x = torch.randn(1, 1, 6, HEAD_DIM)
    out = apply_rope(x, cos, sin)
    print(f'  in  {tuple(x.shape)}   ->   out {tuple(out.shape)}')
    for t in (0, 3):
        tag = 'unchanged (cos=1, sin=0)' if t == 0 else 'rotated'
        print(f'    x[0,0,{t},:4]   = {[round(v, 6) for v in x[0, 0, t, :4].tolist()]}')
        print(f'    out[0,0,{t},:4] = {[round(v, 6) for v in out[0, 0, t, :4].tolist()]}   <- {tag}')
    print('  norms preserved:')
    print('    in : ' + '  '.join(f'{x[0, 0, t].norm():.6f}' for t in range(6)))
    print('    out: ' + '  '.join(f'{out[0, 0, t].norm():.6f}' for t in range(6)))


def trace_pairing_trap():
    section('TRACE 3  which implementation mistake actually breaks relativity?')
    dim = 8
    c, s = build_rope(dim, CONTEXT)
    torch.manual_seed(0)
    q, k = torch.randn(dim), torch.randn(dim)

    def adjacent(x, pos):
        cc, ss = c[pos], s[pos]
        e, o = x[..., 0::2], x[..., 1::2]
        return torch.stack((e * cc - o * ss, e * ss + o * cc), -1).flatten(-2)

    def half_split(x, pos):
        cc, ss = c[pos], s[pos]
        e, o = x[..., :dim // 2], x[..., dim // 2:]
        return torch.stack((e * cc - o * ss, e * ss + o * cc), -1).flatten(-2)

    def identity(x, pos):
        return x

    print('  spread of the score over absolute positions at a fixed gap (0 = relative):')
    for name, rq, rk in (('adjacent pairing, q and k both rotated', adjacent, adjacent),
                         ('half-split pairing, q and k both rotated', half_split, half_split),
                         ('rotate q ONLY, k untouched', adjacent, identity)):
        vals = [(rq(q, m) * rk(k, m + 3)).sum().item() for m in (0, 5, 12, 20)]
        spread = max(vals) - min(vals)
        print(f'    {name:42s} spread={spread:9.6f}  '
              f'{"relative OK" if spread < 1e-5 else "BROKEN, silent"}')


def trace_block_shapes():
    section('TRACE 4  Block.forward shape flow (B=2, T=5)')
    torch.manual_seed(0)          # the q.k^T table below is only reproducible with a fixed seed
    cfg = dict(vocab=2048, width=128, heads=4, depth=4, context=256, pos_mode='rope')
    model = student.build_model(cfg).eval()
    batch, length = 2, 5
    heads, head_dim = cfg['heads'], cfg['width'] // cfg['heads']
    block = model.blocks[0]

    def show(label, tensor):
        print(f'  {label:38s} {tuple(tensor.shape)}')

    x = torch.randn(batch, length, cfg['width'])
    show('x', x)
    normed = block.norm1(x)
    show('norm1(x)', normed)
    packed = block.qkv(normed)
    show('qkv(normed)          Linear(128,384)', packed)
    reshaped = packed.view(batch, length, 3, heads, head_dim)
    show('.view(B,T,3,heads,head_dim)', reshaped)
    permuted = reshaped.permute(2, 0, 3, 1, 4)
    show('.permute(2,0,3,1,4)  -> [3,B,H,T,D]', permuted)
    q, k, v = permuted
    show('unpack q (k, v identical)', q)
    rq = apply_rope(q, model.rope_cos, model.rope_sin)
    rk = apply_rope(k, model.rope_cos, model.rope_sin)
    show('apply_rope(q)   (v NOT rotated)', rq)
    attended = F.scaled_dot_product_attention(rq, rk, v, is_causal=True)
    show('scaled_dot_product_attention', attended)
    back = attended.transpose(1, 2).reshape(batch, length, cfg['width'])
    show('.transpose(1,2).reshape(B,T,width)', back)
    print()
    print('  causal mask actually applied (raw q.k^T, B=0 H=0):')
    raw = rq[0, 0] @ rk[0, 0].T
    for i in range(length):
        print('    ' + '  '.join(f'{raw[i, j]:7.3f}' if j <= i else '  MASK ' for j in range(length)))


def trace_modes():
    section('TRACE 5  the three arms differ only in position wiring')
    cfg = dict(vocab=2048, width=128, heads=4, depth=4, context=256)
    keys = {}
    for mode in student.POS_MODES:
        model = student.build_model(dict(cfg, pos_mode=mode))
        keys[mode] = set(model.state_dict())
        print(f'  {mode:8s} params={sum(p.numel() for p in model.parameters()):>9,}  '
              f'pos_embedding={"yes" if "pos.weight" in keys[mode] else "no ":3s}  '
              f'rope_buffers={hasattr(model, "rope_cos")}  state_dict={len(keys[mode])}')
    print()
    print(f'  none and rope share the same key set : {keys["none"] == keys["rope"]}')
    print(f'  learned differs from none by         : {keys["learned"] - keys["none"]}')


def main():
    trace_build_rope()
    trace_relative_property()
    trace_apply_rope()
    trace_pairing_trap()
    trace_block_shapes()
    trace_modes()


if __name__ == '__main__':
    main()
