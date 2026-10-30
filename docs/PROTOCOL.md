# Frozen evaluation protocol

This is the only setup that may be quoted as a comparison. It is the rolling-origin design already used for seasonal naive, SARIMA and the local GRU. Fixed-split commands (`fl_experiment.local_gru`, `fl_experiment.run_fl_simulation`, `fl_experiment.centralized_baseline`) are software smoke tests. They early-stop or score the whole validation year, and they cannot write a result for the thesis.

FH-PFL has a draft definition in [methods/fh_pfl.md](methods/fh_pfl.md). It has no implementation and no scores. FH-PFL v2 ([methods/fh_pfl_v2.md](methods/fh_pfl_v2.md)) is scored on the validation origins only, with this protocol's frozen settings; see [the validation note](experiments/2026-10-30-fh-pfl-v2-validation.md). FH-PFL v3 ([methods/fh_pfl_v3.md](methods/fh_pfl_v3.md)) is also scored on validation only; see [its note](experiments/2026-10-30-fh-pfl-v3-validation.md). The GST-only confirmation amendment has not been made yet.

## Data and target

- Panel: official GST, E-Way Bills and CPI, CPI base 2012, assumed publication lag of 45 days. The loader must be called with the assumed-vintage opt-in.
- Target: `target_gst_excluding_cess_crore`, one month ahead, in INR crore.
- These runs are retrospective sensitivity results. Strict real-time history still has no complete rows, because original release vintages are not archived.

## Origins

- Season length 12. Input window 6. Early-stopping holdout 6. At least 24 training months.
- One series per state. A window is used only when every month in it exists. Nothing is interpolated.
- For origin month `m`, fitting and scaling use only months before `m`. The six months immediately before `m` are the holdout. Month `m` is predicted once. It is not used to stop training or to choose a federated round.
- Validation origins: 2024-04 through 2025-03. Test origins start at 2025-04. Test labels stay unused until `fl_experiment.evaluate_test` is run against a frozen `selection.json`.

## Architecture, frozen before the comparison runs

| Choice | Value |
|---|---|
| Model | GRU, hidden size 16, dropout 0.1, linear head |
| Optimiser | Full-batch Adam, learning rate 0.01, weight decay 1e-4 |
| Local and pooled GRU | At most 500 epochs, patience 30, restore the best holdout weights |
| Federated round | 5 local epochs on the fit windows |
| Federated cap | At most 20 rounds, patience 5 on holdout equal-state RMSE |
| Seeds | 0, 1 and 2 for every neural method |

Seasonal naive and SARIMA stay the deterministic reference already recorded. Their orders are not searched again.

## What validation is allowed to choose

The selection score is the mean, across the three seeds, of equal-state RMSE in INR crore. MAE, MAPE and MASE are reported and are not used to pick a setting.

1. Feature set, `gst_only` or `all`, chosen from the local GRU. Every other neural method then uses that same set.
2. FedProx strength `mu` in {0.01, 0.1, 1}. `mu = 0` is FedAvg, not a second candidate.
3. APFL mixing step size in {0.1, 1.0}, starting from `alpha = 0.5`.

The chosen values are written to `selection.json` with a hash of the folds. The test command refuses to run if that file is missing or the hash differs.

## Methods

- **Local GRU.** One model per state, retrained from the seed at each origin.
- **Pooled GRU.** One model on the eligible states' fit windows for that origin. One scaler, fit on those pooled fit rows.
- **FedAvg.** Sample-weighted average of the clients' parameters, one client per eligible state. The kept round is the best holdout round.
- **FedProx.** The same average, with client loss `MSE + (mu / 2) ||w - w_global||^2`. `mu = 0` is required to match FedAvg.
- **APFL.** Deng, Kamani and Mahdavi (2020), adapted as described in [methods/apfl.md](methods/apfl.md). The scored model is `alpha * local + (1 - alpha) * global`. `alpha` moves on the client's holdout loss and stays in `[0, 1]`.

## Two execution paths for the federated methods

FedAvg, FedProx and APFL can run in one process (`fl_experiment.compare`, `fl_experiment.evaluate_test`) or as Flower 1.11 clients, one Flower simulation per origin (`fl_experiment.flower_rolling`). Both paths call the same client update in `fl_experiment/federated.py`. In the Flower path, the FedProx strategy only sends `proximal_mu`, and the client applies the penalty. APFL keeps its mixing weight in the node's context state.

Either path may be quoted, under two checks. With dropout 0, the two paths must give the same predictions on a synthetic origin. With the frozen settings on the real validation year, their equal-state RMSE must agree within the seed standard deviation. Dropout draws differ between processes, so exact agreement is not expected there.

## The test year after 29 October 2026

April 2025 through March 2026 was scored once, on 29 October 2026, under `docs/experiments/2026-10-29-selection.json`. Any later scoring of those months, by a new method or by a different execution path, is a second look. It must be labelled that way, and it may not change a frozen setting.
