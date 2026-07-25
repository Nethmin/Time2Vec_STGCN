# Finalized STGCN + Time2Vec temporal model

A clean, self-contained replication of the thesis's **final** model: the STGCN
traffic-forecasting backbone augmented with a learnable-frequency **Time2Vec**
temporal encoding (the thesis contribution). All experimental scaffolding from
the research codebase — NMF adaptive adjacency, graph fusion, discrete ToD/DoW
embeddings, gated fusion, alternative optimizers/schedulers/losses, EMA — has
been removed. Only the final recipe remains, with the few knobs that matter
left configurable.

## Layout

```
stgcn_final/
├── main.py                 # train + evaluate (finalized recipe baked in)
├── make_dashboard.py       # build the interactive prediction dashboard (viva)
├── model/
│   ├── layers.py           # STGCN blocks + ReparamTime2Vec + TemporalEncoding
│   └── models.py           # STGCNChebGraphConv
├── script/
│   ├── dataloader.py       # adjacency / velocity loading + sliding windows
│   ├── earlystopping.py    # best-val checkpointing
│   └── utility.py          # GSO construction + evaluation metrics
├── data/pemsd7-m/          # adj.npz + vel.csv (the thesis dataset)
└── requirements.txt
```

## The model

- **Backbone:** STGCN with ChebGraphConv (Ks=3, Kt=3), 2 ST-Conv blocks, GLU
  activation, symmetric normalized-Laplacian GSO, dropout 0.5.
- **Temporal encoding:** `ReparamTime2Vec` at the **daily** and **weekly**
  scales, each producing `d_t2v` features (1 linear trend + `d_t2v - 1` sine
  "waves"), concatenated and linearly projected to `d_emb=16` channels, then
  concatenated onto the velocity input channel.
  - Frequencies are reparameterized as `ω = ω_base · exp(log_scale)` with
    `ω_base` fixed at **integer harmonics** (1…k cycles per period) and
    `log_scale` initialized to 0 — so waves start at integer harmonics and the
    network learns multiplicative deviations.

## The recipe (defaults in `main.py`)

| | |
|---|---|
| dataset | PeMSD7-M (228 sensors, 5-min) |
| input / horizon | `n_his=12` / `n_pred=3` (15 min) |
| temporal | Time2Vec, `d_t2v=6`, `d_emb=16` |
| loss | MSE (L2, faithful to the original STGCN paper) |
| optimizer | AdamW, lr=0.004, weight_decay=1e-3 |
| frequency params | lr × 10, no weight decay; phases: no weight decay |
| schedule | cosine (`t_max=60`) + 2-epoch linear warmup |
| batch / dropout / clip | 256 / 0.5 / grad-norm 5 |
| selection | best **validation** loss (early stopping, patience 10) |

**Reported result** (test, PeMSD7-M, 15-min, mean ± std over 5 seeds):
**MAE 2.207 ± 0.018**, RMSE 3.951 ± 0.012, MAPE 5.23%.

## Usage

```bash
# Final model, default config (15-min horizon)
python main.py

# A specific seed (thesis reports mean ± std over seeds 42, 1, 7, 13, 21)
python main.py --seed 7 --save_path dt2v6_s7

# Longer horizons
python main.py --n_pred 6      # 30 min
python main.py --n_pred 9      # 45 min

# Vary the number of Time2Vec waves (d_t2v = #waves + 1)
python main.py --d_t2v 4
python main.py --d_t2v 8

# Backbone-only baseline (no temporal encoding)
python main.py --no-temporal_emb
```

Training selects the model by validation loss and reloads that checkpoint for
the final test-set report. A `*_history.csv` of per-epoch train/val loss is
written next to the checkpoint.

## Prediction dashboard (for the viva)

`make_dashboard.py` builds a **single offline HTML file** of predicted-vs-actual
speed for **all 228 sensors** at the 15 / 30 / 45-min horizons:

```bash
python make_dashboard.py                 # -> dashboard/predictions_dashboard.html
python make_dashboard.py --max_hours 72  # lighter file: show only the first 3 days
```

Open the file in any browser (no server, no internet). Controls:

- **Sensor dropdown** — switch between all 228 sensors
- **Horizon toggle** — All / 15 / 30 / 45 min (legend shows that sensor's MAE)
- Hover for values, drag to zoom, double-click to reset
- A **per-sensor MAE overview** bar chart (all sensors, grouped by horizon) sits
  underneath

By default it loads the best-seed (s7) checkpoints from
`../stgcn_temporal/Final model selection/` (the clean model is checkpoint-
compatible). Point it at your own checkpoints with `--ckpt15/--ckpt30/--ckpt45`.

> Only the PeMSD7-M dataset is bundled (the thesis dataset). The code keeps the
> calendar settings for metr-la / pems-bay in `DATASET_CFG`; drop their
> `adj.npz` + `vel.csv` into `data/<name>/` to run them.
