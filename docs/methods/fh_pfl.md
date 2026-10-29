# FH-PFL — draft specification (#21)

> **Status: draft. Not implemented, not tuned, not scored.** The definition choices below were agreed on 29 October 2026 (see [Agreed decisions](#agreed-decisions)). This is a proposed design, not an established or validated algorithm, and nothing here is a claim of novelty; see the [literature matrix](../literature/fh_pfl_matrix.md).
>
> **Closest prior work.** FedAP (Lu et al., 2022) already aggregates with a fixed similarity matrix computed from data statistics, using a self-weight λ plus \((1-\lambda)\) times normalised weights on the others, and keeps some layers private. HeurFedAMP (Huang et al., 2021) uses the same self-weight-plus-softmax form, with model-parameter similarity. FH-PFL differs in its similarity input (economic revenue profiles recomputed at each origin) and in choosing \(\alpha_i\) per state on the holdout. It is a test of whether that input helps, not a new aggregation rule.

FH-PFL (Federated Heterogeneity-Aware Personalized Learning) changes how much each state shares. FedAvg gives every state the same global model. FH-PFL gives each state its own aggregate of the other states' encoders, weighted by how similar their training-period revenue behaviour is, and keeps a private prediction head.

## Hypothesis

States benefit more from sharing temporal representations with economically similar states than from sharing one uniform global model.

The test is the matched comparison of FH-PFL against its own uniform-aggregation ablation (A1 below), with everything else held fixed. A win over FedAvg alone would not test this hypothesis, because FH-PFL also adds a private head and a local blend.

## Model

The existing `RevenueModel` already has the split:

- encoder \(g_\theta\): `model.gru`, a one-layer GRU. With the frozen GST-only input and hidden size 16 it has 912 parameters;
- head \(h_\phi\): `model.head`, dropout and a linear layer, 17 parameters.

\[ \hat y_{i,t} = h_{\phi_i}\big(g_{\theta_i}(X_{i,t})\big) \]

The forecast is made in each state's own scaled target space and inverted with the training-only scaler, exactly as for the other methods. The head is a linear read-out, so "private head" means a private scale-and-offset on the shared representation, not a separate network. A larger head would be a second change and is out of scope for the first version.

## Profiles

For each state \(i\) and origin \(m\), the profile \(z_i^{(m)}\) is computed from the target series on fit months only. Fit months are months strictly before the six-month holdout that precedes \(m\). The holdout, the origin month and anything later are never used.

| Component | Definition on the fit months |
|---|---|
| Growth | Mean of 12-month log changes, \(\log y_t - \log y_{t-12}\) |
| Volatility | Standard deviation of month-on-month log changes |
| Seasonality | Seasonal strength, as in `heterogeneity._seasonal_strength`: the share of linearly detrended \(\log y\) variance explained by month-of-year means |

Each component is z-scored across the states present at that origin. Revenue level is excluded, so absolute size cannot drive similarity. A level-inclusive profile is an optional sensitivity check, not part of the method. The heterogeneity note found that level relates to which method wins, so this choice should be stated, not hidden.

The server receives three summary statistics per state. The series are public, so this is not a disclosure problem here, but it is information FedAvg does not send. No privacy claim follows from either.

## One round

Notation: \(r\) is the round, \(\tilde\theta_i^r\) is the encoder state \(i\) receives at the start of round \(r\), \(\phi_i^r\) is its private head and \(N\) is the number of states at the origin.

**Initialisation.** Every state starts from the same seeded initial model: \(\tilde\theta_i^1 = \theta^0\) and \(\phi_i^1 = \phi^0\). The similarity weights are fixed for the whole origin and run over all \(N\) states, including \(i\) itself:

\[ a_{ij} = \frac{\exp(-\lVert z_i - z_j \rVert^2 / \tau)}{\sum_{k=1}^{N} \exp(-\lVert z_i - z_k \rVert^2 / \tau)} \]

Because \(\lVert z_i - z_i \rVert = 0\), the self-weight \(a_{ii}\) is the largest in each row, and it grows as \(\tau\) shrinks. After the blend below, state \(i\)'s total weight on its own encoder is \(\alpha_i + (1 - \alpha_i)\,a_{ii}\). So \(\tau\) and \(\alpha_i\) both move the self-share, and the results must report the effective self-share, not \(\alpha_i\) alone. States are weighted equally regardless of sample count. Histories differ by only three months, from 61 to 64, so a sample-weighted variant would differ very little; it is listed as a check, not a headline ablation.

**Local step (client \(i\)).** Starting from \((\tilde\theta_i^r, \phi_i^r)\), run the same five full-batch Adam epochs on the fit windows as FedAvg, FedProx and APFL, minimising

\[ L_i(\theta, \phi) = \mathrm{MSE}_i(\theta, \phi) + \frac{\mu}{2} \lVert \theta - \tilde\theta_i^r \rVert^2 \]

The proximal term acts on the encoder only. The result is \((\theta_i^{r+1}, \phi_i^{r+1})\). The head stays on the client, and only \(\theta_i^{r+1}\) is sent.

**Aggregation (server).** For each state the server forms \(\bar\theta_i^{r+1} = \sum_{j=1}^{N} a_{ij}\, \theta_j^{r+1}\) from the unblended encoders the clients sent, and returns it to state \(i\).

**Blend and \(\alpha_i\) (client).** The client already holds its own \(\theta_i^{r+1}\) and \(\phi_i^{r+1}\). For each \(\alpha\) in the grid \(A = \{0.25, 0.5, 0.75\}\) it forms

\[ \tilde\theta_i^{r+1}(\alpha) = \alpha\, \theta_i^{r+1} + (1 - \alpha)\, \bar\theta_i^{r+1} \]

and scores \(h_{\phi_i^{r+1}}(g_{\tilde\theta_i^{r+1}(\alpha)})\) on its six-month holdout. It keeps \(\alpha_i^{r+1}\), the grid value with the lowest holdout RMSE; ties go to the smaller \(\alpha\), meaning more sharing. The chosen \(\tilde\theta_i^{r+1}\) starts the next round. Other states never see \(\tilde\theta_i\) or \(\alpha_i\), because aggregation uses the unblended \(\theta_j\). So each state's choice of \(\alpha\) does not couple to the others within a round.

The holdout is the inner chronological split: the six months before the origin. The origin month and the test year are never used. The same six points also choose the round, so \(\alpha_i\) and the round are selected on one small sample. This is a known overfitting risk of the agreed design. The fixed-\(\alpha\) ablation A3 measures it.

**Scoring.** The model a state would deploy after round \(r\) is \(h_{\phi_i^{r+1}}(g_{\tilde\theta_i^{r+1}})\), with the blended encoder and no extra head training. Its holdout RMSE is the one already computed for \(\alpha_i^{r+1}\). Round selection is unchanged from the frozen protocol: at most 20 rounds, stop after 5 rounds without improvement in equal-state holdout RMSE, and keep the best round. That round's model forecasts the origin month.

## Hyperparameters and budget

Everything not listed here is the frozen selection (`docs/experiments/2026-10-29-selection.json`): GST-only input, window 6, hidden 16, dropout 0.1, learning rate 0.01, weight decay 1e-4, seeds 0, 1 and 2, five local epochs, 20 rounds and patience 5.

| Setting | Proposal |
|---|---|
| \(\tau\) | Grid \(\{1, 3, 10\}\). With three standardised components, the expected squared distance between two random states is about 6, so 1 is sharp and 10 is nearly uniform |
| \(\alpha_i\) | Chosen per state, per round, from \(\{0.25, 0.5, 0.75\}\) on the holdout, as above. It is not tuned across origins |
| \(\mu\) | Fixed at the frozen FedProx value 0.01, not re-tuned |

That gives three configurations, one per \(\tau\). Each is scored over three seeds on the 12 validation origins and selected by mean equal-state RMSE, the same rule used for FedProx and APFL. The grid is fixed before any run and is not widened after the results are seen.

## Ablations

Each ablation starts from the selected \(\tau\) and changes one thing:

| ID | Change | Question it answers |
|---|---|---|
| A1 | Uniform weights \(a_{ij} = 1/N\), the limit \(\tau \to \infty\) | Does profile similarity matter? This is the hypothesis test |
| A2 | No private head: the head is aggregated and blended like the encoder | Does keeping the head local matter? |
| A3 | Fixed sharing coefficient \(\alpha_i = 0.5\) for every state and round, with no holdout choice | Does choosing \(\alpha_i\) per state help, or does it overfit the six holdout points? |
| A4 | Sample-count-weighted \(a_{ij} \propto n_j \exp(\cdot)\) | Check only. Little difference is expected with 61 to 64 months per state |

A1 to A3 run on the validation origins with the same seeds and budget. The baselines are already scored on the same origins: local GRU, FedAvg, FedProx and APFL.

## Evaluation and claims

1. **Development:** validation origins 2024-04 to 2025-03 only. That covers the grid, ablations and debugging.
2. **Test year 2025-04 to 2026-03:** this year has already been opened. Any FH-PFL number on it is reported as a second look, in a separate table, and is not used to choose anything.
3. **Confirmation period.** Complete rows end in March 2026 because the CPI base-2012 series stops there. April to August 2026 have GST labels and every input except CPI. The frozen selection is GST-only, so those five months, 160 state forecasts, can be scored without CPI. This needs a protocol amendment: a GST-only panel that does not require `complete_features`, with a new hash recorded before the run. No method, including the baselines, has been scored on these months. Five months is a small sample. Treat it as confirmation of direction, not of effect size.
4. **Decision rule, fixed before the confirmation run:** the hypothesis counts as supported only if FH-PFL has lower equal-state RMSE than A1 on both the validation origins and the confirmation period, in the mean over seeds and in at least two of the three seeds. Otherwise the thesis reports that profile-weighted sharing did not help on this data. Per-state differences are reported either way.
5. FH-PFL is compared with SARIMA and seasonal naive as well. SARIMA is the strongest method on both years so far, and a federated method that does not beat it must say so.

## Implementation outline (after agreement)

- `federated.py`: `profiles(clients, settings)`, `similarity_weights(z, tau)`, and an `fhpfl_origin` that reuses `local_update` with the proximal term on `gru.*` keys only, keeps per-state server-side aggregates, and does the client-side \(\alpha_i\) choice.
- Flower: a strategy whose `configure_fit` sends each client its own \(\bar\theta_i\). The private head, the client's last \(\theta_i\) and \(\alpha_i\) persist in `context.state`, as APFL already does with \(\alpha\). Agreement with the in-process path is checked as in `docs/experiments/2026-10-29-flower-agreement.md`.
- Tests:
  - With uniform weights, \(\alpha_i = 0\) and a shared head, the encoder reproduces FedAvg.
  - No profile reads a holdout or origin month.
  - Heads never leave the client.
  - The \(\alpha_i\) choice reads only holdout rows.

## Agreed decisions

Agreed on 29 October 2026:

1. **Neighbour set:** the aggregate includes the state itself, as in the original formula. \(\tau\) and \(\alpha_i\) both move the self-share, and the effective self-share \(\alpha_i + (1 - \alpha_i)\,a_{ii}\) is reported.
2. **Scored model:** the blended encoder with the private head, without a head refit.
3. **\(\alpha_i\):** chosen per state, per round, from \(\{0.25, 0.5, 0.75\}\) on the six-month holdout. A3 measures the overfitting risk.
4. **Confirmation period:** April to August 2026 under a GST-only protocol amendment, with its hash recorded before any method is scored on those months.
5. **\(\mu\):** fixed at 0.01.
