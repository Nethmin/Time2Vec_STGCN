"""Export learned Time2Vec parameters from a locally trained checkpoint.

Frequencies are reported as cycles per day or five-weekday week, and phases
in radians. Trend coefficients and projection weights are also exported.

Usage:
    python inspect_temporal.py --ckpt h15_s42.pt
"""

import argparse
import math
import os


def scale_table(sd, prefix, period_name):
    import numpy as np
    import pandas as pd

    log_scale = sd[f'{prefix}.log_scale'].numpy()
    phase = sd[f'{prefix}.phase'].numpy()
    omega_base = sd[f'{prefix}.omega_base'].numpy()
    harmonic = np.round(omega_base / (2 * math.pi)).astype(int)        # integer init 1..k
    omega = omega_base * np.exp(log_scale)
    cycles = omega / (2 * math.pi)                                     # cycles per period
    df = pd.DataFrame({
        'wave': np.arange(1, len(log_scale) + 1),
        'harmonic_init': harmonic,
        f'cycles_per_{period_name}': np.round(cycles, 4),
        'log_scale': np.round(log_scale, 4),
        'phase_rad': np.round(phase, 4),
        'omega': np.round(omega, 4),
    })
    slope = float(sd[f'{prefix}.linear.weight'].reshape(-1)[0])
    bias = float(sd[f'{prefix}.linear.bias'].reshape(-1)[0])
    return df, slope, bias


def main():
    p = argparse.ArgumentParser(description='Inspect a locally trained Time2Vec checkpoint')
    p.add_argument(
        '--ckpt', type=str, required=True,
        help='locally trained checkpoint with temporal encoding enabled')
    p.add_argument('--outdir', type=str, default=os.path.join('runs', 'temporal_encoding'))
    args = p.parse_args()

    if not os.path.isfile(args.ckpt):
        p.error(f'Checkpoint not found: {args.ckpt}. '
                'No pretrained weights are bundled; train a model with main.py first.')

    import numpy as np
    import pandas as pd
    import torch

    sd = torch.load(args.ckpt, map_location='cpu', weights_only=True)
    os.makedirs(args.outdir, exist_ok=True)
    tag = os.path.splitext(os.path.basename(args.ckpt))[0]

    print(f'\nLearned temporal encoding  ({os.path.basename(args.ckpt)})')
    print('=' * 64)

    for prefix, period in [('temporal_emb.t2v_daily', 'day'),
                           ('temporal_emb.t2v_weekly', 'week')]:
        df, slope, bias = scale_table(sd, prefix, period)
        print(f'\n[{period.upper()}LY scale]  linear trend:  slope={slope:+.4f}  bias={bias:+.4f}')
        print(df.to_string(index=False))
        out = os.path.join(args.outdir, f'{tag}_{period}.csv')
        df.to_csv(out, index=False)

    W = sd['temporal_emb.projection.weight'].numpy()
    b = sd['temporal_emb.projection.bias'].numpy()
    proj = pd.DataFrame(W, columns=[f'in_{i}' for i in range(W.shape[1])])
    proj.insert(0, 'bias', np.round(b, 4))
    proj.to_csv(os.path.join(args.outdir, f'{tag}_projection.csv'), index=False)

    print('\n' + '=' * 64)
    print(f'CSVs written to {args.outdir}/ :')
    print(f'  {tag}_day.csv, {tag}_week.csv, {tag}_projection.csv')


if __name__ == '__main__':
    main()
