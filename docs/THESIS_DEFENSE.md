# Findings for the thesis defense

Chronological record of what the project has established, in the order the evidence was produced. Each finding says what can be defended, the number or check behind it, and the limit that has to be stated in the same breath.

The defendable empirical comparison is the matched validation table in [experiments/2026-10-29-protocol-validation.md](experiments/2026-10-29-protocol-validation.md). The earlier rolling-origin table remains the record of how the seasonal naive, SARIMA and the first local GRU were scored. Everything before those tables is data definition, protocol, or an exploratory run. The original proposal in [PROPOSAL.md](PROPOSAL.md) is the starting question, not a result. Work on `fl_initial_experiment` (synthetic panels, welfare-leakage models, approximate differential privacy) is not on this branch and is not evidence for this thesis.

The test period, April 2025 through March 2026, was scored once after the validation choices were frozen. It was not used to pick a feature set, a FedProx strength, or an APFL step. Neural numbers are means across seeds 0, 1 and 2.

## What can be said in one minute

Monthly state GST can be studied as a forecasting problem on official data, with one client per state and a locked evaluation. On a retrospective sensitivity panel, a per-state SARIMA is the accuracy reference on both the validation year (RMSE 228 against a seasonal-naive 427) and the test year (288 against 392). FedAvg does not beat a local GRU. APFL was the best neural method on validation, close to the seasonal naive, and on the test year it falls behind that naive forecast (509 against 392). No privacy guarantee has been tested.

## 1. The question was narrowed before any model was trained

The archived proposal asked for privacy-preserving GST forecasting and welfare-leakage detection, with claims about encrypted updates and a privacy–accuracy trade-off. The roadmap in [#30](https://github.com/amit-tutu22/FL-State-level-Forecasting/issues/30) keeps state-level monthly GST forecasting and personalized federated learning. Issue [#21](https://github.com/amit-tutu22/FL-State-level-Forecasting/issues/21) places welfare-leakage detection outside this thesis unless a supervisor approves it separately.

**Defend:** the thesis question is next-month state GST, with each state as one client. **State alongside it:** privacy wording in the proposal is aspirational. No privacy mechanism has been implemented or measured on this branch.

## 2. Official series exist, and their definitions are fixed — 29 October 2026

Source audit: [audits/2026-10-29.md](audits/2026-10-29.md). Pipeline: [REAL_DATA_PIPELINE.md](REAL_DATA_PIPELINE.md). Acquisition was repeated the same day and reproduced the audit counts.

| Series | Coverage actually in the files | Rows |
|---|---|---:|
| State domestic gross GST | July 2017–August 2026, 110 months, 10 workbooks | 4,264 |
| E-Way Bills, three directions kept separate | July 2018–August 2026, 98 months, 9 workbooks | 3,702 |
| CPI General Combined | January 2013–September 2026 | 6,618 |

Checks that support using the GST levels:

- All 110 monthly workbook totals reconcile with their component rows.
- 70 state-months in the September and December 2023 official PDFs match the workbooks within INR 0.51 crore.
- CPI bases 2012 and 2024 are stored separately. The 2024 back series is All India only, so it is not assigned to states. April–May 2020 are missing in the 2012 state CPI and were left missing.

**Defend:** the modelling target is domestic gross GST, in INR crore, gross of refunds, and it is not the amount a state keeps after IGST settlement. The historically consistent series is `target_gst_excluding_cess_crore` (CGST + SGST + IGST). The published total includes cess through FY2024–25 and then drops the cess column, so that total is not one series across the whole history.

**State alongside it:** post-settlement receipts are a different question and are not this target. Dataful was not used. The official workbooks are the acquisition route.

## 3. A reference month is not a publication date

The downloaded files are current revisions. The original monthly release that contained each historical value has not been archived, so the pipeline refuses to pretend those values were known at the time.

- Strict availability, the default, produces **0 complete historical rows** (3,877 GST rows joined, none fully usable).
- The labelled sensitivity mode assumes a value is public 45 days after month-end. On CPI base 2012 that yields **2,858 complete rows**. On CPI base 2024 it yields 595. The client loader rejects this mode unless the run opts in.

Every modelling number below uses the 2,858-row panel. CPI base 2012 is the longer overlap (2,867 state-months at the reference-month level; base 2024 has 700).

**Defend:** the project can build a state-month panel, and it can show that a leakage-safe backtest is currently empty. **State alongside it:** the 2,858-row results are retrospective sensitivity results. They still contain revision bias. They are not evidence of real-time forecast skill. A true backtest needs an original-release registry, which the code accepts and which has not been filled.

## 4. Not every label in the file is a client

Geography that changed, or that the source does not define cleanly, is quarantined rather than renamed or added together. That covers the October 2019 Jammu & Kashmir split month, Ladakh before it existed, the old-base CPI Jammu & Kashmir series after the split, and Daman & Diu / Dadra & Nagar Haveli labels after the January 2020 merger. All India, Other Territory, and centre-only jurisdictions stay in the audit tables and are not clients.

When the panel is cut for modelling (at least 24 training windows, a validation year, and a held-out year), **32 states and union territories remain**. Four do not:

- Arunachal Pradesh: no usable training window on this panel.
- Undivided Jammu & Kashmir: 7 training windows.
- Legacy Daman & Diu, and legacy Dadra & Nagar Haveli: 10 training windows each.

In the later rolling-origin run those four have no complete month from April 2024 onward, so they are absent there too.

**Defend:** the client count is a result of coverage and boundary rules, not a decision to use 28 or 36 clients. **State alongside it:** quarantined rows are not a claim that those territories have no GST. They are a claim that the source definition is not yet safe to model.

## 5. The forecast protocol was fixed before the comparison

Recorded in [fl_experiment/README.md](../fl_experiment/README.md) and the rolling-origin note.

- Horizon is one month. Season length is 12.
- For target month `m`, inputs are values already public at the forecast origin, which is the last day of month `m-1`. The target itself is never an input in the same month.
- The synthetic generator's `sgst_share` and `cgst_share` columns are functions of the target. They are not used. Synthetic data remains a software smoke test.
- Each state owns all of its rows. There is no resampled or overlapping client.
- Scaling is fit on that state's training slice only.
- A six-month GRU window is used only when all six months exist, so the April–May 2020 CPI gap is not crossed or filled.
- Validation origins are April 2024–March 2025. Test origins from April 2025 are written to `folds.json` with `scored: false`. This run scored 384 validation origins and stored 384 test origins, and scored none of the test origins.

**Defend:** model choice did not use the final test year, and the three models below saw the same origins. **State alongside it:** this is one validation year. Rolling origins inside that year are done. A second locked test evaluation, multiple seeds, and a pre-registered configuration (#25) are not done.

## 6. The first real-data run showed the pipeline works, and is not the comparison

[experiments/2026-10-29-real-data-smoke.md](experiments/2026-10-29-real-data-smoke.md). One GRU per state, trained on all months before April 2024, early-stopped on the whole validation year. Seed 0. Hidden size 16.

| Model | Inputs | Validation RMSE | MAPE % |
|---|---|---:|---:|
| Local GRU | GST, six E-Way Bill features, CPI | 407.3 | 10.33 |
| Local GRU | GST history only | 367.9 | 9.49 |
| FedAvg GRU, 30 rounds, all inputs | same as the first row | 516.7 | 12.74 |

GST-only had the lower MAPE in 22 of 32 states. FedAvg's validation RMSE was still falling at round 30 (670 at round 1, 517 at round 30). Each FedAvg round is five local epochs, so its budget is shorter than the early-stopped local models.

**Defend:** official data can be trained end to end, one client per state, in Flower 1.11.1, and the run records per-state errors. **Do not defend these RMSE figures as the result.** Early stopping saw the whole year being scored, FedAvg was not trained to a comparable budget, and there was no seasonal naive or SARIMA on this split. The next section replaces this table.

Lakshadweep, Andaman & Nicobar, Mizoram, Tripura and Meghalaya had the worst percentage errors. Lakshadweep's monthly collection is about 1 crore, so a small crore error is a large percentage. The all-inputs Lakshadweep model stopped at epoch 1.

## 7. On a common rolling origin, SARIMA is the reference and the GRU does not beat a seasonal naive

[experiments/2026-10-29-rolling-origin.md](experiments/2026-10-29-rolling-origin.md). For each of the 12 validation months, every model is refit using only earlier months. The GRU early-stops on the six months before the origin, not on the origin. SARIMA picks one order per state by AIC on data before April 2024, then only refits parameters. The seasonal naive is the same month last year. Wall time was 52 seconds. No SARIMA fit failed, so no origin fell back to the naive forecast.

| Model | RMSE | MAE | MAPE % | MASE, pooled | MASE, equal across states |
|---|---:|---:|---:|---:|---:|
| Seasonal naive | 427.2 | 381.3 | 18.85 | 1.00 | 1.00 |
| SARIMA | 228.1 | 185.9 | 17.28 | 0.49 | 0.75 |
| Local GRU, all inputs | 479.7 | 317.3 | 11.76 | 0.83 | 1.03 |
| Local GRU, GST history only | 440.2 | 312.6 | 11.18 | 0.82 | 1.00 |

MASE is this model's MAE divided by the seasonal-naive MAE on the same origins. A value of 1 is a tie with "same month last year." Pooled MASE treats every origin equally, so large states dominate. Equal-state MASE is the average of the 32 state scores.

What this supports:

- **SARIMA is the baseline a later federated model has to beat.** It has a lower RMSE than the seasonal naive in 27 of 32 states. Twenty-nine states selected the airline order `(0,1,1)(0,1,1,12)`. Goa, Rajasthan and Tamil Nadu selected `(1,1,1)(0,1,1,12)`. The pooled MASE of 0.49 against the equal-state MASE of 0.75 means the crore gains sit mostly in the large states.
- **The local GRU is not more accurate than a seasonal naive for a typical state.** GST-only equal-state MASE is 1.00. Its MAPE is lower than the naive MAPE in 21 of 32 states, and its RMSE is lower in 11. It has a lower MAPE than SARIMA in 8 states. Its RMSE, 440, is worse than the naive RMSE, 427.
- **The extra series did not help on this design.** GST-only is slightly better than GST plus E-Way Bills plus CPI on every aggregate in both the fixed split and the rolling origin. Monthly sample size does not grow when columns are added. This is one seed and one lag assumption, so it is a result to carry into the feature ablation (#26), and it is already evidence against claiming that those series help.
- **Percentage error and crore error answer different questions.** Lakshadweep's seasonal-naive MAPE is 286% and SARIMA's is 304%, on a level of about 1 crore. The GST-only GRU is at 38% there. That percentage gain is invisible in RMSE, which is decided by Maharashtra-scale states. A defense should report both, and should say which one the loss function targeted. The GRU is trained in log space, which is closer to a percentage goal. SARIMA is fit in crore.

**State alongside it:** one seed, untuned GRU width, dropout, learning rate and window, and SARIMA orders chosen by AIC rather than by validation error. A tuned GRU could move. The test year has not been opened, so this is the validation conclusion, not the final number.

## 8. What has been built for federation, and what has not been shown

Flower starts, with the interface current for version 1.11.1. Each eligible state is one client. A 30-round FedAvg run completes on the real panel. The server records sample-weighted, equal-state, and per-state errors every round. Separate operating-system processes can also connect as clients. That is an engineering result (issue [#18](https://github.com/amit-tutu22/FL-State-level-Forecasting/issues/18)).

The matched scientific comparison is in section 9. FedAvg, FedProx and APFL now share the rolling origins with the local and pooled GRUs. FH-PFL still has no update rule. There is no communication-cost measurement.

**Defend:** the simulation is reproducible and the state partition is disjoint. **State alongside it:** public GST, E-Way Bill and CPI series are already public. Training on them in a Flower simulation does not create a privacy guarantee, and the thesis must not say that it does.

## 9. On the frozen protocol, SARIMA still leads and APFL is the best neural method

[experiments/2026-10-29-protocol-validation.md](experiments/2026-10-29-protocol-validation.md) and the freeze file [experiments/2026-10-29-selection.json](experiments/2026-10-29-selection.json). The rules in [PROTOCOL.md](PROTOCOL.md) were written down before this run. The feature set was chosen from the local GRU, then every other neural method used it. The score was equal-state RMSE, averaged over seeds 0, 1 and 2.

GST history alone beat GST plus E-Way Bills plus CPI (446.9 against 470.4). FedProx was best at `mu = 0.01` and still worse than FedAvg (474.6 against 469.9); `mu = 1` was worse again (521.0). APFL with step size 1.0 was the best neural setting, at 416.3 ± 13.1 crore. The pooled GRU was the worst, at 585.3.

**Defend:** under this protocol, sharing one model across states does not help, a stronger FedProx penalty does not help, and the personalized mixture is the neural method that comes closest to the seasonal naive on the validation year. **State alongside it:** SARIMA's validation RMSE is 228. APFL beats SARIMA there in only five small states. Maharashtra still dominates the crore error. The test year, scored afterwards, is in section 10.

## 10. The test year does not confirm the neural methods

[experiments/2026-10-29-protocol-test.md](experiments/2026-10-29-protocol-test.md). The command refused to start unless the frozen selection file matched the fold hash. Settings were not searched again. Origins run from April 2025 through March 2026.

| Method | Test RMSE |
|---|---:|
| Seasonal naive | 392.0 |
| SARIMA | 287.6 |
| APFL, step 1 | 509.1 ± 29.7 |
| FedAvg | 536.5 ± 15.3 |
| Local GRU, GST only | 539.8 ± 5.4 |
| FedProx, μ = 0.01 | 540.3 ± 20.1 |
| Pooled GRU | 615.9 ± 30.7 |

**Defend:** the held-out year still ranks SARIMA first, and it still ranks a single pooled GRU last. **State alongside it:** APFL's validation advantage over the seasonal naive disappears. Its test RMSE is 509 against 392, and its equal-state MASE is 1.21. It beats SARIMA in one state, Lakshadweep, where both errors are about 2 crore. The local GRU has the lowest MAPE (13.97%) and a worse RMSE than the seasonal naive. Maharashtra still decides the crore error.

## 11. The federated numbers do not depend on how they were executed

[experiments/2026-10-29-flower-agreement.md](experiments/2026-10-29-flower-agreement.md). FedAvg, FedProx and APFL were first computed in one process. They were then rerun as Flower 1.11 clients, with one Flower simulation per origin. The client update is the same code in both paths. Flower's FedProx strategy only sends `mu`, and the client adds the penalty. With dropout switched off, both paths give the same predictions on a synthetic origin. With the frozen settings on the real panel, they agree within the seed spread. On validation, Flower gives APFL 421.0 against 416.3, and FedAvg 472.1 against 469.9. On the test year, labelled a second look, APFL is 510.2 against 509.1.

**Defend:** the federated results are Flower results, and the conclusions in sections 9 and 10 hold for either path. **State alongside it:** the Flower re-score of the test year is a second look at months already reported. It checks the execution path, and it does not replace the first test run.

## Claims that are not available for the defense

| Claim | Status |
|---|---|
| The GRU, FedAvg, or APFL forecasts GST better than SARIMA | SARIMA leads on validation (228) and on the test year (288). The best neural test RMSE is APFL at about 509. |
| A neural method beats the seasonal naive | APFL was close on validation and behind on the test year (509 against 392). |
| E-Way Bills or CPI improve the forecast | The local GRU is worse when those series are added. The other methods then used GST history only. |
| FedAvg or FedProx beats a local GRU | On validation both are worse. On the test year FedAvg is within a seed's spread of the local GRU and still far from SARIMA. |
| The system is privacy-preserving, encrypted, or differentially private | No such mechanism is on this branch. |
| Welfare leakage can be detected | Out of scope for this roadmap. |
| Results from the synthetic `fl_initial_experiment` branch | Different data, different model, not merged here. |
| A real-time backtest | The test year was scored, but on the assumed 45-day lag. Strict real-time history has zero complete rows. |

## What is still open, in the order the roadmap asks for it

1. The literature matrix and a definition of FH-PFL that can be falsified (#21), then the semester review (#22). FH-PFL is not implemented.
2. Ablations of personalization, lag length and the extra series (#26). Another pass over the test year is not one of those ablations.
3. Original-release vintages, if a real-time claim is going to be made. Until then the 45-day assumption stays in every table.
