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

The saved dashboard uses seed 7 at each horizon. Its single-seed metrics are separate from the five-seed averages in these tables.
