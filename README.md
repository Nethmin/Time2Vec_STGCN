# STGCN + Time2Vec for Traffic Forecasting

Final-model implementation of the HND Data Science research project **A Time2Vec Temporal Encoding for Multi-Horizon Traffic Forecasting with Spatio-Temporal Graph Neural Networks**.

The model adds explicit daily and weekday calendar context to a Spatio-Temporal Graph Convolutional Network (STGCN). Two continuous Time2Vec streams start at integer harmonics and learn positive frequency adjustments during training. The encoding is concatenated with the speed input before the STGCN backbone.

This repository includes training code, the PeMSD7-M dataset, historical result tables, and an offline prediction dashboard. **Pretrained checkpoints are not included.** Training creates local weights for evaluation and for rebuilding the dashboard.

## Model

- Fixed distance-based road graph, with a scaled symmetric normalized Laplacian.
- Two ST-Conv blocks with Chebyshev graph convolution (`Ks=3`), temporal convolutions (`Kt=3`), GLU, layer normalization, and dropout.
- Two Time2Vec streams: daily and weekly. Here a week means the five recorded weekdays.
- Each stream produces one linear trend and five sine components (`d_t2v=6`). Their 12 features are projected to 16 channels (`d_emb=16`) and concatenated with speed, giving 17 input channels.
- Frequencies follow `omega = omega_base * exp(log_scale)`, starting at harmonics 1 through 5 cycles per period. Frequencies and phases are learned.
- An input of 12 five-minute observations predicts one future speed per sensor. Separate models handle 15-, 30-, and 45-minute horizons.

The temporal encoding module contains 232 parameters. The complete model contains 262,761 registered parameters; the full parameter increase over the backbone-only model is larger than the encoding module because concatenation also widens the first convolution.

## Reported research results

Historical test results on PeMSD7-M, reported as mean +/- sample standard deviation across seeds **42, 1, 7, 13, 21**:

| Horizon | MAE (mph) | RMSE (mph) | MAPE (%) |
|---|---:|---:|---:|
| 15 minutes | 2.207 +/- 0.018 | 3.951 +/- 0.012 | 5.23 +/- 0.02 |
| 30 minutes | 2.845 +/- 0.048 | 5.233 +/- 0.024 | 7.16 +/- 0.09 |
| 45 minutes | 3.216 +/- 0.115 | 5.911 +/- 0.104 | 8.18 +/- 0.27 |

The capacity `d_t2v=6` was selected by the lowest mean validation loss across seeds in a sweep over `{4,6,8}`. Each run uses its best-validation checkpoint for testing. The small capacity differences overlap with seed variation.

The original per-run and summary CSVs are in [results/](results/README.md). These are recorded research results, not newly executed benchmarks of this cleaned implementation. The dashboard shows a single seed and is separate from the five-seed averages above.

## Setup

Use Python 3.12, the version used during development, and create a fresh virtual environment. Run commands from the directory containing `main.py`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On Linux or macOS, activate with `source .venv/bin/activate`. PyTorch uses CUDA when available and otherwise runs on CPU. Install a PyTorch build appropriate for your hardware; see the [official installation guide](https://pytorch.org/get-started/locally/).

## Train and evaluate

Train one model for each horizon, using distinct output names:

```bash
python main.py --seed 42 --n_pred 3 --save_path h15_s42
python main.py --seed 42 --n_pred 6 --save_path h30_s42
python main.py --seed 42 --n_pred 9 --save_path h45_s42
```

Each command saves a local `.pt` checkpoint and a `*_history.csv`, reloads the checkpoint with the best validation loss, and prints test metrics. MAPE is printed as a fraction; multiply by 100 to express it as a percentage. Weights and training histories are excluded from Git.

The defaults implement this training recipe:

| Setting | Value |
|---|---|
| Dataset / split | PeMSD7-M; chronological train/validation/test 70/15/15 |
| Scaling | Per-sensor StandardScaler fitted on training data only |
| Encoding | `d_t2v=6`, `d_emb=16`; daily and five-weekday streams |
| Optimizer / loss | AdamW / MSE in standardized space |
| Learning rate / weight decay | 0.004 / 0.001 |
| Frequency parameters | 10x base learning rate; no weight decay |
| Phase parameters | Base learning rate; no weight decay |
| Schedule | Two-epoch warmup followed by cosine annealing (`t_max=60`) |
| Batch / dropout / gradient clipping | 256 / 0.5 / norm 5 |
| Maximum epochs / early stopping | 100 / patience 10 |

To study capacity or train the backbone-only baseline:

```bash
python main.py --d_t2v 4 --save_path capacity4_s42
python main.py --d_t2v 8 --save_path capacity8_s42
python main.py --no-temporal_emb --save_path baseline_s42
```

## View the prediction dashboard

Open [dashboard/predictions_dashboard.html](dashboard/predictions_dashboard.html) in a browser. The saved HTML works offline without Python or checkpoints. It contains predictions from the historical seed-7 runs, with controls for all 228 sensors and the three forecast horizons. It is a demonstration, not the seed-averaged result table.

After training the three default-capacity models above, rebuild the dashboard using your local weights:

```bash
python make_dashboard.py --ckpt15 h15_s42.pt --ckpt30 h30_s42.pt --ckpt45 h45_s42.pt --max_hours 72
```

The generated file is `runs/predictions_dashboard.html`. Omit `--max_hours` to include the entire test period, or use `--out` to choose another destination. All three checkpoint arguments are required; the builder has no dependency on the original experiment folders.

## Inspect learned temporal features

```bash
python inspect_temporal.py --ckpt h15_s42.pt
```

This prints learned frequencies, phases, and trend coefficients, and writes CSVs to `runs/temporal_encoding/`. The input must be a model trained with temporal encoding enabled. Use `--outdir` to change the output location.

The existing [temporal_encoding/](temporal_encoding/) CSVs are small historical exports from the 15-minute seed-7 model. They are provided for interpretation; they do not contain the complete model weights.

## Data and reproducibility

`data/pemsd7-m/` contains the speed matrix `vel.csv` and the fixed adjacency `adj.npz` for 228 sensors. The benchmark records five-minute speeds on weekdays in May-June 2012. The dataset was obtained through the [original STGCN data distribution](https://github.com/VeritasYin/STGCN_IJCAI-18/blob/master/data_loader/PeMS-M.zip) and carried forward from the reference implementation. The code license does not establish separate licensing terms for third-party data.

The existing research loader uses pandas' default CSV-header handling: the first numeric row is consumed as column names, leaving 12,671 observations and split lengths 8,871/1,900/1,900. Temporal indices start at zero for the first retained row, producing a five-minute offset relative to the raw-file calendar assumption. The window builder also omits the last otherwise valid window. These inherited behaviors are retained to preserve the historical pipeline; changing them requires a new experiment.

The cleaned implementation constructs the reparameterized modules directly. The original research notebooks constructed ordinary Time2Vec modules before replacing them and performed an additional evaluation pass over the shuffled training loader each epoch. Those procedures consume random numbers differently, so a new run with the same seed is not guaranteed to reproduce the exact historical weights or metric row. Dependency versions are lower bounds, not an exact environment lock.

Only PeMSD7-M was evaluated in the final thesis. Calendar settings for METR-LA and PeMS-BAY remain available, but their datasets are not bundled here.

## Repository layout

```text
main.py                       Train and evaluate the final model
model/                        STGCN backbone and reparameterized Time2Vec
script/                       Data loading, graph operators, metrics, checkpointing
data/pemsd7-m/                Benchmark speeds and fixed adjacency
results/                      Historical per-seed and capacity result tables
make_dashboard.py             Build a dashboard from locally trained weights
inspect_temporal.py           Inspect locally trained encoding parameters
dashboard/                    Saved offline demonstration
temporal_encoding/            Historical encoding-parameter exports
requirements.txt              Dependencies
LICENSE                       Upstream LGPL 2.1 license text
NOTICE.md                     Attribution and description of modifications
```

## Attribution

The STGCN backbone and data-loading code are derived from [hazdzz/STGCN](https://github.com/hazdzz/STGCN), implementing the work of Bing Yu, Haoteng Yin, and Zhanxing Zhu. This HND research adds the harmonic-seeded, reparameterized daily/weekly Time2Vec encoding and the accompanying training configuration and analysis utilities.

- Yu, Yin, and Zhu, [Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting](https://arxiv.org/abs/1709.04875), IJCAI 2018.
- Kazemi et al., [Time2Vec: Learning a Vector Representation of Time](https://arxiv.org/abs/1907.05321), 2019.

See [NOTICE.md](NOTICE.md) for provenance and [LICENSE](LICENSE) for the upstream GNU Lesser General Public License, version 2.1.
