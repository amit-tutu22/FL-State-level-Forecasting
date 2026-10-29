# FH-PFL v2 — federated top-down shares (specification, #21)

> **Status: specification agreed on 30 October 2026. Not yet scored.** This is a second design, separate from the [FH-PFL v1 draft](fh_pfl.md), which is unchanged. It is a proposed design, not an established or validated algorithm.
>
> **Prior work, and what may be claimed.** The first v2 idea, coherence constraints solved by ADMM, turned out to be the mechanism of Tan et al. (2021). That idea was dropped; see the [literature review](../literature/review.md#hierarchical-coherence-novelty-check-for-fh-pfl-v2). This design is the top-down approach instead: forecast the total, then split it with forecast proportions.
>
> - Top-down forecasting with proportions is classical: Gross and Sohl (1990); Athanasopoulos, Ahmed and Hyndman (2009).
> - Learning the total and the proportions end to end in one network is Das et al.'s Dirichlet Proportions Model (UAI 2023), which is centralized.
>
> A search on 30 October 2026 found no federated version, in which each region keeps a private model and the proportions are normalised through a secure sum. That is a search result, not proof of absence. The most this design may claim is "a federated form of learned top-down proportions, applied to Indian state GST". It must not be called a new forecasting principle.

## Idea

Every state's revenue is its share of the all-India total: \(y_{i,t} = s_{i,t} N_t\).

- **The national series is public.** The server forecasts it from public data alone.
- **Each state models only its own share.** It uses a fully private model and its own data, and no model parameters ever leave a state.
- **The states are coupled through the normalisation.** The shares must add up to one, and the joint training objective feeds back the states' combined share error. Both are passed only as sums across states.
- **Personalization:** every state has its own model.
- **Collaboration:** the states' share forecasts must be consistent with one another and with the national forecast.

## Threat model and what leaves a state

- **Public:**
  - \(N_t\), defined as the sum of the federation states' monthly GST excluding cess. GSTN publishes state and all-India collections, and with this definition the forecasts are coherent by construction.
  - CPI, which v2 does not use.
- **Treated as private:**
  - each state's own series;
  - its model, scaler and forecasts.

  In reality, state GST collections are published too. The experiment *simulates* a setting in which they are not, and the thesis must say so.
- **What each state uploads per round:** two scalars per fit month, \(e^{g_{i,t}}\) and \(e_{i,t}\) (defined below), one scalar per holdout month, and one at the forecast month. No weights or gradients are sent.
- **Assumption: secure summation.** The server sees only the sums \(Z_t\) and \(E_t\) across states. Without secure summation, the pair \((e^{g_{i,t}}, e_{i,t})\) would reveal \(s_{i,t}\), and so \(y_{i,t}\). The experiment computes the sums in-process and implements no cryptography. No differential-privacy or formal privacy claim is made.

## Federation months

\(N_t\) and the shares exist only in months where every federation state has a row. In the frozen fold file, the 32 scored states are the same at every validation origin. Their gaps differ:
- every state misses July and August 2020;
- Manipur, Nagaland and Puducherry also miss September 2020;
- Lakshadweep misses April to June 2021.

Share targets, the national input column, the input windows and the national SARIMA all use the months common to every state. A window never crosses a gap, as for the other GRUs. The national SARIMA uses the contiguous run of months before the origin, as the state SARIMA does. If the set of states changes between origins, the run stops with an error rather than redefining \(N_t\).

## Model

Each state \(i\) keeps its own `RevenueModel` with the frozen architecture: GRU hidden size 16, dropout 0.1, window 6.

- **Input** for target month \(t\): the state's frozen GST-only window (its lagged GST feature and the month encoding), plus one public column per window step. That column is \(\log\) of the national lagged GST feature, \(\sum_j\) `feature_gst_excluding_cess_crore`\(_{j}\), built from the same rows and so with the same availability as the state's own input.
- **Output:** a normalised score, decoded to a log-share score \(g_{i,t} = \sigma_i \cdot \mathrm{out} + m_i\). Here \(m_i\) and \(\sigma_i\) are the mean and standard deviation of \(\log s_{i,t}\) over the state's fit months.
- **Shares:**

\[ \hat s_{i,t} = \frac{e^{g_{i,t}}}{Z_t}, \qquad Z_t = \sum_{j=1}^{N} e^{g_{j,t}} \quad \text{(secure sum)} \]

\[ \hat y_{i,t} = \hat s_{i,t}\,\hat N_t \]

The forecasts add up exactly to \(\hat N_t\) by construction.

## National anchor

\(\hat N_m\) is a one-step SARIMA forecast of \(N_t\) from months before \(m\), using the same code as the state SARIMA baseline:
- the order is chosen by AIC from the full order grid on months before `val_start`, then frozen;
- the model is refit at each origin;
- if a fit fails, it falls back to the seasonal naive, and the fallback is counted.

The anchor is fixed in advance and not tuned on any FH-PFL result.

## Objective and one round

Let \(F\) be the fit months shared by every state at origin \(m\), and \(H\) the six holdout months before \(m\). The joint objective is the equal-state mean squared log-share error:

\[ J(\theta_1,\dots,\theta_N) = \frac{1}{N}\sum_{i=1}^{N} \frac{1}{|F|}\sum_{t\in F} e_{i,t}^2, \qquad e_{i,t} = \log\hat s_{i,t} - \log s_{i,t} = g_{i,t} - \log Z_t - \log s_{i,t} \]

Because \(Z_t\) depends on every state's score, the gradient for state \(i\) involves the others only through two sums:

\[ \frac{\partial J}{\partial g_{i,t}} \propto e_{i,t} - \hat s_{i,t}\, E_t, \qquad E_t = \sum_{j} e_{j,t} \]

**Round \(r\):**

1. **Score.** Each state computes \(g_{i,t}\) for \(t \in F\) with its current \(\theta_i\).
2. **First secure sum.** The server receives \(Z_t = \sum_j e^{g_{j,t}}\) and broadcasts \(c_t = \log Z_t\).
3. **Second secure sum.** Each state computes \(e_{i,t} = g_{i,t} - c_t - \log s_{i,t}\). The server receives \(E_t = \sum_j e_{j,t}\) and broadcasts it.
4. **Local step.** Each state runs the frozen 5 full-batch Adam epochs on the surrogate

\[ S_i(\theta) = \frac{1}{|F|}\sum_{t\in F}\Big[\big(g_{i,t}(\theta) - c_t - \log s_{i,t}\big)^2 - 2E_t\, e^{\,g_{i,t}(\theta) - c_t}\Big] \]

   with \(c_t\) and \(E_t\) held at their round-start values. At the start of the round, \(\nabla S_i\) equals \(N \cdot \partial J / \partial \theta_i\), so a round is a stale-gradient step on the shared objective. Weight decay is applied as for the other methods.
5. **Round score.**
   - Each state scores its holdout months \(t \in H\), with one secure sum of \(e^{g}\) per holdout month.
   - The holdout forecast is \(\hat y_{i,t} = \hat s_{i,t} N_t\), using the actual \(N_t\), which is known before the origin. This scores share accuracy without needing an extra SARIMA fit per holdout month.
   - The round's score is the equal-state holdout RMSE in INR crore.
   - At most 20 rounds, stop after 5 without improvement, keep the best round, as in the frozen protocol.
6. **Forecast.** One last secure sum gives \(Z_m\), and \(\hat y_{i,m} = \hat s_{i,m}\,\hat N_m\).

There is no new hyperparameter. Everything is the frozen selection in `docs/experiments/2026-10-29-selection.json`: seeds 0, 1 and 2, learning rate 0.01, weight decay 1e-4, 5 local epochs, 20 rounds, patience 5. Each state starts from the same seeded initialisation, as in FedAvg.

## Baselines and ablations

All of these use the same validation origins and seeds.

| ID | What changes | What it isolates |
|---|---|---|
| A1 | **Independent shares.** Each state fits \(g_{i,t}\) to \(\log s_{i,t}\) on its own, with no \(Z\) and no \(E\), and forecasts \(e^{g_{i,m}} \hat N_m\). Nothing is federated | Whether the federated coupling adds anything beyond a private share model plus the public anchor |
| A2 | **Post-hoc normalisation.** A1's scores, normalised by one secure sum at the forecast month only | Whether coupling *during training* (the \(E_t\) term) adds anything beyond normalising |
| A3 | **Classical top-down.** \(\hat N_m\) times the state's mean share over its last 12 fit months. No GRU, no federation | Whether learning the shares beats historical proportions |
| A4 | Seasonal-naive national anchor instead of SARIMA | Sensitivity to the anchor |
| A5 | No national input column | Whether the public national history helps the share model |
| Oracle | The actual \(N_m\) in place of \(\hat N_m\) | Upper bound only, reported separately and never used for a decision |

These are compared with the frozen validation numbers already on record, as equal-state RMSE in INR crore:

| Method | RMSE |
|---|---|
| SARIMA | 228 |
| APFL | 416 |
| Seasonal naive | 427 |
| Local GRU | 447 |
| FedAvg | 470 |

FH-PFL v1 is not implemented, so it is not a comparator yet.

## Decision rule (fixed before any run)

1. **Federation claim.** Federated coupling counts as helping only if v2 has lower equal-state RMSE than **both A1 and A2**. That must hold on the validation origins and on the confirmation period, in the seed mean and in at least two of three seeds.
2. **Usefulness claim.** Report v2 against the local GRU, A3 and SARIMA. If v2 does not beat SARIMA, say so plainly.
3. **Test year (2025-04 to 2026-03).** It has already been opened, so it is a second look only and chooses nothing.
4. **Confirmation (April–August 2026).** GST and the GST-only features exist for these months, but CPI does not. Scoring them needs the GST-only protocol amendment, which must be written and hashed before any method is run there.

## Tests (before any real run)

- `S_i`'s gradient at round start equals the autograd gradient of \(J\), to within 1e-6 on a small synthetic panel.
- The forecasts add up to \(\hat N_m\) exactly at every origin.
- Client uploads contain only per-month scalars, never tensors of model parameters.
- No origin-month target or feature is read by the fit, the holdout or the national anchor.
- With the \(E_t\) term removed and \(Z\) frozen at 1, a round reduces to A1's local update.

## Implementation outline (after agreement)

- **New `fl_experiment/shares.py`:**
  - the national series and its input column, built at each origin from the fold file;
  - share targets and scalers;
  - the SARIMA anchor, reusing `fit_sarima` and `select_state_order` from `baselines.py` on a national pseudo-state;
  - the round loop, the ablations and prediction rows in the existing format.
- **Flower:** clients return their per-month scalars as the fit payload, and a strategy sums them and broadcasts \(c_t\) and \(E_t\) in the fit config. The two paths are checked to agree at dropout 0, as for FedAvg.
- **Runner:** writes validation-origin results for v2 and A1–A5 to `fl_experiment/results/`, plus a note under `docs/experiments/`.

## Choices agreed on 30 October 2026

1. **Definition of \(N_t\):** the sum of the federation states, so coherence is exact. The published all-India total was rejected because it includes territories outside the panel and the centre's share, so the panel shares would not add up to one.
2. **Public national input column:** included, with A5 as its ablation.
3. **Loss scale:** log-share squared error. RMSE in crore is used only for round selection and reporting.
