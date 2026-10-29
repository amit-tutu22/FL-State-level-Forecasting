# FedAvg, FedProx and APFL through Flower, 29 October 2026

The matched comparison computed FedAvg, FedProx and APFL in one process. This note runs the same frozen settings as Flower 1.11.1 clients, one Flower simulation per origin, and checks that the numbers agree. Nothing is searched again. The settings come from [2026-10-29-selection.json](2026-10-29-selection.json): GST-only inputs, FedProx `mu = 0.01`, APFL step 1.0, and seeds 0, 1 and 2.

The client update is the same code in both paths (`fl_experiment/federated.py`). In Flower:

- the FedProx strategy only sends `proximal_mu`, and the client adds the penalty;
- APFL keeps its mixing weight in the node's context state between rounds;
- the server keeps the round with the lowest equal-state holdout RMSE, and stops selecting clients after five rounds without improvement.

Unit tests with dropout 0 give the same predictions in both paths, to a relative tolerance of 1e-5, for:

- FedAvg;
- FedProx with `mu = 0`;
- APFL.

With dropout 0.1, the random draws differ between processes, so the real-panel check uses the seed spread as its tolerance.

```sh
python -m fl_experiment.flower_rolling --split val \
    --selection docs/experiments/2026-10-29-selection.json \
    --reference fl_experiment/results/real-2026-10-29/protocol/metrics.json \
    --source panel --allow-assumed-vintages --panel data/reports/lagged45-2012/panel.csv
```

## Validation year, April 2024 – March 2025

32 states, 12 origins and three seeds. 108 Flower simulations took 29 minutes.

| Method | Flower RMSE | In-process RMSE | Gap | Within seed spread |
|---|---:|---:|---:|---|
| FedAvg | 472.1 ± 8.5 | 469.9 ± 7.7 | +2.2 | yes |
| FedProx, μ = 0.01 | 473.4 ± 9.0 | 474.6 ± 13.7 | −1.2 | yes |
| APFL, step 1 | 421.0 ± 15.0 | 416.3 ± 13.1 | +4.8 | yes |

Equal-state RMSE, INR crore, mean ± standard deviation across seeds. The Flower MAE, MAPE and MASE are also within a few percent of the in-process run. For example, APFL MAPE is 10.57 against 10.50.

The validation ranking is unchanged:

- APFL is ahead of FedAvg;
- FedProx at `mu = 0.01` is level with FedAvg;
- both are behind the GST-only local GRU (446.9) and far behind SARIMA (228.1).

The median kept round was 3. The median number of rounds run before patience stopped training was 8. The 20-round cap was not the limit.
