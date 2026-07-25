"""Extract the LEARNED temporal-encoding parameters from a trained checkpoint.

The temporal encoding is two ReparamTime2Vec modules (daily + weekly) followed
by a linear projection. For each scale the learned quantities are:

  * frequencies  -- omega = omega_base * exp(log_scale); reported as CYCLES per
                    period (cycles/day for the daily scale, cycles/week for the
                    weekly scale). omega_base is fixed at integer harmonics
                    1..k, so a learned cycle near an integer means the wave
                    stayed close to its harmonic init.
  * phases       -- phase offset (radians) of each sine wave.
  * linear trend -- slope (w0) and bias (b0) of the non-periodic component.

The projection (2*d_t2v -> d_emb) mixes the daily+weekly features into the
channels fed to the backbone; its matrix is saved to CSV but not pretty-printed.

Usage:
    python inspect_temporal.py                       # 15-min final model (dt2v6_s7)
    python inspect_temporal.py --ckpt <path.pt>      # any checkpoint
"""

import argparse
import math
import os

import numpy as np
import pandas as pd
import torch

DEFAULT_CKPT = os.path.join('..', 'stgcn_temporal', 'Final model selection', 'dt2v6_s7.pt')


def scale_table(sd, prefix, period_name):
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
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, default=DEFAULT_CKPT)
    p.add_argument('--outdir', type=str, default='temporal_encoding')
    args = p.parse_args()

    sd = torch.load(args.ckpt, map_location='cpu')
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

    # projection matrix (2*d_t2v -> d_emb)
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
