# Historical research results

These CSVs are unchanged copies of the final research result tables. They preserve the numerical evidence for the thesis without distributing trained checkpoints or the earlier experiment workspace.

| File | Contents |
|---|---|
| [results_dt2v_sweep.csv](results_dt2v_sweep.csv) | 15-minute capacity sweep: dimensions 4, 6, and 8, each across five seeds. Filter `d_t2v=6` for the selected model's 15-minute results. |
| [summary_dt2v.csv](summary_dt2v.csv) | Original rounded means and sample standard deviations for that capacity sweep. |
| [results_best_h30.csv](results_best_h30.csv) | Five runs of the selected model at the 30-minute horizon (`n_pred=6`). |
| [results_best_h45.csv](results_best_h45.csv) | Five runs of the selected model at the 45-minute horizon (`n_pred=9`). |

All final runs use PeMSD7-M, `d_emb=16`, MSE loss, and seeds `42,1,7,13,21`. The selected Time2Vec dimension is `d_t2v=6`. Architecture selection uses mean best validation loss across seeds; test metrics are reported after reloading each run's best-validation checkpoint.

## Columns and units

- `best_val` and `val_mean`: MSE in per-sensor standardized space.
- `test_MAE` / `MAE_mean`: mean absolute error in miles per hour.
- `test_RMSE` / `RMSE_mean`: root mean squared error in miles per hour.
- `test_MAPE` / `MAPE_mean`: a fraction, so `0.0523` means approximately `5.23%`.
- Standard-deviation columns use sample standard deviation across seeds (`ddof=1`).
- `best_epoch` and `n_epochs`: selected epoch and total completed training epochs.
- `secs`: elapsed time recorded during the original run; it is not a hardware-independent speed benchmark.

The README's headline results are rounded aggregates of these tables. The cleaned code has the same final architecture and hyperparameters, but its random-number sequence differs from the original notebook procedure. These files should therefore be described as historical research results, not as a guarantee that every new training run yields an identical metric row.

## Reproducing the historical runs

The existing research loader uses pandas' default CSV-header handling. The first numeric row of `vel.csv` is consumed as column names, leaving 12,671 observations and chronological split lengths of 8,871/1,900/1,900. Calendar indices begin at zero for the first retained row, giving a five-minute offset relative to the raw-file calendar assumption. The window builder also omits the last otherwise valid window. These behaviors remain in the code to preserve the historical pipeline; correcting them requires a new experiment.

The cleaned implementation constructs the reparameterized Time2Vec modules directly. The original notebooks first constructed ordinary Time2Vec modules and then replaced them. They also evaluated the shuffled training loader once more per epoch. These differences consume random numbers differently, so an identical seed does not guarantee identical historical weights or a matching metric row. Dependencies are not pinned to an exact environment lock.

The thesis's static prediction figure was produced by the original prediction exporter using `header=None`, a training split of 8,872 rows, and the corresponding scaler and calendar indices. That export path differs from the headline evaluation pipeline. Treat the figure as a historical illustration, not as the source of the table's metrics.

The thesis and presentation compare the final model with published STGCN results. Those reference numbers are not a baseline rerun with the same final batch size, optimizer configuration, and seed set, so they do not establish a controlled measurement of the encoding's effect.

Only PeMSD7-M was evaluated in the final thesis. Calendar settings for METR-LA and PeMS-BAY remain available in the code, but their datasets are not bundled.

## README figures

The figures are extracted directly from the author's thesis and presentation. They have not been regenerated from new training runs.

| Figure | Source |
|---|---|
| [Model architecture](../assets/figures/architecture.png) | Research presentation, slide 9; original embedded PNG. |
| [Learned Time2Vec components](../assets/figures/learned_time2vec.png) | Thesis, PDF page 31 (printed page 30); original embedded PNG showing daily and weekly components. |
| [Prediction examples](../assets/figures/predictions.png) | Thesis, PDF page 30 (printed page 29); original embedded PNG showing network-average speeds. |
