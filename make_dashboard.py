"""Build a self-contained interactive HTML dashboard of the finalized model's
predicted-vs-actual speeds for ALL sensors, at the 15 / 30 / 45-min horizons.

For each horizon it loads the matching best-seed checkpoint into the clean
finalized model, runs the held-out test set, inverse-transforms to mph, and
writes one offline HTML file (Plotly embedded — no server, no internet needed):

    dashboard/predictions_dashboard.html

Controls in the page:
    * Sensor dropdown  (all 228 sensors)
    * Horizon toggle   (All / 15 / 30 / 45 min)
    * A per-sensor MAE overview bar chart underneath

Usage:
    python make_dashboard.py                  # uses the seed-7 (best) checkpoints
    python make_dashboard.py --max_hours 72   # trim the timeline shown
"""

import argparse
import os

import numpy as np
import torch
import plotly.graph_objects as go

from main import get_parameters, data_preparate
from model import models

# Best-seed (s7) checkpoints from the model-selection run, one per horizon.
FMS = os.path.join('..', 'stgcn_temporal', 'Final model selection')
DEFAULT_CKPTS = {
    3: os.path.join(FMS, 'dt2v6_s7.pt'),
    6: os.path.join(FMS, '30 min horizon', 'best_h30_s7.pt'),
    9: os.path.join(FMS, '45 minute horizon', 'best_h45_s7.pt'),
}
HORIZON_MIN = {3: 15, 6: 30, 9: 45}
HORIZON_COLOR = {3: '#1f77b4', 6: '#ff7f0e', 9: '#2ca02c'}


def run_horizon(n_pred, ckpt, seed):
    """Return (hours, actual[T,N], pred[T,N]) on the original mph scale."""
    args, device, blocks = get_parameters(
        ['--n_pred', str(n_pred), '--d_t2v', '6', '--seed', str(seed), '--temporal_emb'])
    n_vertex, zscore, _, _, test_iter = data_preparate(args, device)

    model = models.STGCNChebGraphConv(args, blocks, n_vertex).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    preds, acts = [], []
    with torch.no_grad():
        for x, tod, dow, y in test_iter:        # test_iter is unshuffled -> order preserved
            yp = model(x, tod, dow).view(len(x), -1)
            preds.append(yp.cpu().numpy())
            acts.append(y.cpu().numpy())
    pred = zscore.inverse_transform(np.concatenate(preds))
    act = zscore.inverse_transform(np.concatenate(acts))

    # Absolute target-step index within the test segment -> hours (5-min steps).
    num = pred.shape[0]
    abs_idx = np.arange(num) + args.n_his + n_pred - 1
    hours = abs_idx * 5.0 / 60.0
    return hours, act.astype(np.float32), pred.astype(np.float32)


def per_sensor_metrics(act, pred, mask_value=1e-5):
    """Per-sensor MAE, RMSE, MAPE(%) over the test set. Arrays are [T, N]."""
    err = np.abs(act - pred)
    mae = err.mean(axis=0)
    rmse = np.sqrt((err ** 2).mean(axis=0))
    mask = np.abs(act) > mask_value
    mape = np.where(mask, err / np.maximum(np.abs(act), mask_value), 0.0)
    mape = (mape.sum(axis=0) / np.maximum(mask.sum(axis=0), 1)) * 100.0
    return mae, rmse, mape


def build_dashboard(results, n_vertex, out_path, max_hours):
    """results: {n_pred: (hours, act, pred, mae, rmse, mape)}."""
    horizons = sorted(results.keys())  # [3, 6, 9]

    def clip(hours, arr):
        if max_hours is None:
            return hours, arr
        m = hours <= (hours[0] + max_hours)
        return hours[m], arr[m]

    s0 = 0  # initial sensor

    # ---- traces: one actual (15-min grid) + one predicted line per horizon ----
    fig = go.Figure()
    h15, a15 = clip(results[3][0], results[3][1])
    fig.add_trace(go.Scatter(
        x=h15, y=np.round(a15[:, s0], 1), name='Actual', mode='lines',
        line=dict(color='black', width=2)))
    for np_ in horizons:
        hrs, pr = clip(results[np_][0], results[np_][2])
        mae = results[np_][3][s0]
        fig.add_trace(go.Scatter(
            x=hrs, y=np.round(pr[:, s0], 1), mode='lines',
            name=f'Pred {HORIZON_MIN[np_]} min (MAE {mae:.2f})',
            line=dict(color=HORIZON_COLOR[np_], width=1.4)))

    # ---- sensor dropdown: restyle y + legend name for all traces ----
    sensor_buttons = []
    for s in range(n_vertex):
        ys = [np.round(clip(results[3][0], results[3][1])[1][:, s], 1).tolist()]
        names = ['Actual']
        for np_ in horizons:
            _, pr = clip(results[np_][0], results[np_][2])
            ys.append(np.round(pr[:, s], 1).tolist())
            names.append(f'Pred {HORIZON_MIN[np_]} min (MAE {results[np_][3][s]:.2f})')
        sensor_buttons.append(dict(
            label=f'Sensor {s}', method='restyle',
            args=[{'y': ys, 'name': names}, [0, 1, 2, 3]]))

    # ---- horizon visibility toggle ----
    vis = {
        'All':    [True, True, True, True],
        '15 min': [True, True, False, False],
        '30 min': [True, False, True, False],
        '45 min': [True, False, False, True],
    }
    horizon_buttons = [dict(label=k, method='restyle', args=[{'visible': v}])
                       for k, v in vis.items()]

    fig.update_layout(
        title='Finalized STGCN + Time2Vec — predicted vs actual speed (PeMSD7-M test set)',
        xaxis_title='Hours into test period (weekday-only, 5-min resolution)',
        yaxis_title='Speed (mph)',
        template='plotly_white', hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(t=120),
        updatemenus=[
            dict(buttons=sensor_buttons, direction='down', showactive=True,
                 x=0.0, xanchor='left', y=1.16, yanchor='top'),
            dict(buttons=horizon_buttons, type='buttons', direction='right',
                 showactive=True, x=0.18, xanchor='left', y=1.16, yanchor='top'),
        ],
        annotations=[
            dict(text='Sensor:', x=-0.0, xref='paper', y=1.20, yref='paper',
                 showarrow=False, xanchor='left'),
            dict(text='Horizon:', x=0.16, xref='paper', y=1.20, yref='paper',
                 showarrow=False, xanchor='right'),
        ],
    )

    # ---- overview: per-sensor MAE across all sensors, grouped by horizon ----
    overview = go.Figure()
    for np_ in horizons:
        overview.add_trace(go.Bar(
            x=np.arange(n_vertex), y=results[np_][3],
            name=f'{HORIZON_MIN[np_]} min', marker_color=HORIZON_COLOR[np_]))
    overall = {np_: results[np_][3].mean() for np_ in horizons}
    overview.update_layout(
        title='Per-sensor MAE across all {} sensors  (mean MAE: {})'.format(
            n_vertex, ', '.join(f'{HORIZON_MIN[k]}min={v:.2f}' for k, v in overall.items())),
        xaxis_title='Sensor index', yaxis_title='MAE (mph)',
        barmode='group', template='plotly_white', margin=dict(t=70))

    # ---- write both figures into one offline HTML ----
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    html_main = fig.to_html(full_html=False, include_plotlyjs=True, div_id='ts')
    html_over = overview.to_html(full_html=False, include_plotlyjs=False, div_id='ov')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(
            '<!DOCTYPE html><html><head><meta charset="utf-8">'
            '<title>STGCN+Time2Vec predictions</title>'
            '<style>body{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#222}'
            'h2{margin:8px 0}.note{color:#666;font-size:13px;margin-bottom:18px}'
            'hr{border:none;border-top:1px solid #ddd;margin:32px 0}</style></head><body>'
            '<h2>Finalized STGCN + Time2Vec — interactive prediction dashboard</h2>'
            '<div class="note">Pick any of the 228 sensors from the dropdown; toggle the '
            'forecast horizon (15 / 30 / 45 min). Hover for values, drag to zoom, '
            'double-click to reset. Fully offline.</div>'
            + html_main + '<hr>' + html_over + '</body></html>')
    return overall


def main():
    p = argparse.ArgumentParser(description='Build the prediction dashboard')
    p.add_argument('--seed', type=int, default=7, help='seed of the checkpoints to load (best=7)')
    p.add_argument('--ckpt15', type=str, default=DEFAULT_CKPTS[3])
    p.add_argument('--ckpt30', type=str, default=DEFAULT_CKPTS[6])
    p.add_argument('--ckpt45', type=str, default=DEFAULT_CKPTS[9])
    p.add_argument('--max_hours', type=float, default=None,
                   help='trim the timeline shown (e.g. 72 = first 3 days); default = full test set')
    p.add_argument('--out', type=str, default=os.path.join('dashboard', 'predictions_dashboard.html'))
    args = p.parse_args()

    ckpts = {3: args.ckpt15, 6: args.ckpt30, 9: args.ckpt45}
    results = {}
    n_vertex = None
    for n_pred, ck in ckpts.items():
        print(f'\n=== horizon {HORIZON_MIN[n_pred]} min (n_pred={n_pred}) | {ck} ===')
        hours, act, pred = run_horizon(n_pred, ck, args.seed)
        mae, rmse, mape = per_sensor_metrics(act, pred)
        results[n_pred] = (hours, act, pred, mae, rmse, mape)
        n_vertex = act.shape[1]
        print(f'  overall: MAE {mae.mean():.3f} | RMSE {rmse.mean():.3f} | MAPE {mape.mean():.2f}%')

    overall = build_dashboard(results, n_vertex, args.out, args.max_hours)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f'\nDashboard written: {args.out}  ({size_mb:.1f} MB)')
    print('Overall test MAE  ->  ' + ', '.join(f'{HORIZON_MIN[k]}min: {v:.3f}' for k, v in overall.items()))


if __name__ == '__main__':
    main()
