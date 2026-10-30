# FH-PFL v3: formal definition and properties (methodology chapter draft, #21)

> This is the formal version of the [v3 specification](fh_pfl_v3.md), written for the methodology chapter.
> - It defines the method and proves four elementary properties.
> - It does **not** prove that v3 beats SARIMA or reaches any RMSE. No such result is available, and the chapter must not imply one.
> - Prior work: v3 extends the classical ARIMA–neural hybrid (Zhang, 2003) and its federated FedAvg form (Allassak, Trichni and Omary, 2026), which is ablation B2 here. See the spec.

## 1. Setting and notation

- **States and series.** \(i = 1,\dots,N\) indexes the federation states, here \(N = 32\). \(y_{i,t} > 0\) is state \(i\)'s GST collection (excluding cess, INR crore) in month \(t\).
- **Inputs.** \(\mathbf{x}_{i,t}\) is the input row available before month \(t\), as defined by the panel's assumed 45-day lag. It holds the lagged GST collection, the six E-Way Bill series and the month encoding \((\sin, \cos)\), all log-transformed where nonnegative. CPI is excluded.
- **Origins and splits.** An origin is a target month \(m\). Its holdout is \(H_m = \{m-6,\dots,m-1\}\). Its fit set \(F_m\) is the target months before \(m-6\) that have a complete 6-month input window and a defined residual (Section 2).
- **Information set.** \(\mathcal{D}_{i,<s}\) denotes state \(i\)'s observations strictly before month \(s\). Everything used for origin \(m\) is a function of \(\{\mathcal{D}_{i,<m}\}_i\).

## 2. SARIMA base and residuals

Each state has a SARIMA order \(\omega_i\), chosen by AIC on \(\mathcal{D}_{i,<\text{2024-04}}\) and then frozen. At origin \(m\), two parameter vectors are estimated by maximum likelihood:

\[
\hat\phi^{R}_{i,m} \text{ on } \mathcal{D}_{i,<m-6} \quad\text{(residual fit)}, \qquad
\hat\phi^{F}_{i,m} \text{ on } \mathcal{D}_{i,<m} \quad\text{(forecast fit)}.
\]

Write \(\hat y^{S}_{i,t}(\phi)\) for the Kalman-filter one-step prediction of \(y_{i,t}\) from \(\mathcal{D}_{i,<t}\) under parameters \(\phi\). Define:

\[
\hat y^{S}_{i,t} =
\begin{cases}
\hat y^{S}_{i,t}(\hat\phi^{R}_{i,m}), & t \in F_m \cup H_m,\\[2pt]
\hat y^{S}_{i,m}(\hat\phi^{F}_{i,m}), & t = m,
\end{cases}
\qquad
r_{i,t} = \log y_{i,t} - \log \hat y^{S}_{i,t}, \quad t \in F_m \cup H_m .
\]

The first 13 months of each contiguous run are dropped while the filter initialises. \(\hat y^{S}_{i,m}\) is the SARIMA baseline's own forecast, so the comparison with SARIMA is like for like. If the forecast fit fails, it falls back to the seasonal naive \(y_{i,m-12}\), and the fallback is counted.

**Lemma 1 (no look-ahead).** Every quantity used at origin \(m\) is measurable with respect to \(\{\mathcal{D}_{i,<m}\}_i\). The holdout residuals \(r_{i,t}\), \(t \in H_m\), are computed with parameters estimated without \(H_m\).

*Proof.*
- \(\hat\phi^{R}_{i,m}\) uses \(\mathcal{D}_{i,<m-6}\), and \(\hat\phi^{F}_{i,m}\) uses \(\mathcal{D}_{i,<m}\).
- A filtered prediction for \(t\) uses \(\mathcal{D}_{i,<t}\) with \(t \le m\).
- The inputs \(\mathbf{x}_{i,t}\) are available before \(t \le m\), by the panel's construction.
- The residual targets use \(y_{i,t}\) only for \(t \le m-1\).
- The second claim holds because \(\hat\phi^{R}_{i,m}\) is estimated on \(\mathcal{D}_{i,<m-6}\) and \(H_m\) starts at \(m-6\). ∎

## 3. Residual model

- **Normalisation.** Residuals are normalised per state on the fit months:

\[ z_{i,t} = (r_{i,t} - \mu_i)/\sigma_i, \qquad \mu_i, \sigma_i \text{ from } \{r_{i,t}\}_{t \in F_m}. \]

- **Network.** \(f_\theta\) is the frozen GRU (hidden size 16, dropout 0.1). It reads the window \(X_{i,t} = (\tilde{\mathbf{x}}_{i,t-5},\dots,\tilde{\mathbf{x}}_{i,t})\). Each \(\tilde{\mathbf{x}}_{i,s}\) is \(\mathbf{x}_{i,s}\) plus the column \(\log \hat y^{S}_{i,s}\), standardised with the state's fit-month means and standard deviations.
- **Decoded residual forecast:** \(\hat r_{i,t}(\theta) = \sigma_i f_\theta(X_{i,t}) + \mu_i\).

## 4. Personalized federated training (APFL)

This is the frozen APFL rule ([apfl.md](apfl.md)), applied to residual targets. Each state keeps local weights \(w_i\) and a mixing weight \(\alpha_i \in [0,1]\), which starts at 0.5. The server keeps the global weights \(\bar w\). The state's model is the mixture

\[ \theta_i = \alpha_i w_i + (1-\alpha_i)\,\bar w . \]

In round \(k = 1,\dots,20\):
1. **Local step.** Five full-batch Adam epochs from \(\bar w\) on \(L_i(w) = |F_m|^{-1}\sum_{t\in F_m}\big(f_w(X_{i,t}) - z_{i,t}\big)^2\). The learning rate is 0.01 and the weight decay \(10^{-4}\).
2. **Mixing step.** \(\alpha_i \leftarrow \Pi_{[0,1]}\big(\alpha_i - \eta\,\partial_{\alpha} L^{H}_i(\theta_i)\big)\), with \(\eta = 1\). Here \(L^H_i\) is the same loss on \(H_m\) and \(\Pi\) clips to the interval.
3. **Aggregation.** \(\bar w \leftarrow \sum_i |F_m^{(i)}|\, w_i \big/ \sum_i |F_m^{(i)}|\).

**Round selection.** Keep the round \(k^\star\) that minimises the equal-state holdout RMSE in crore of the full-correction forecast (\(\lambda = 1\)):

\[ k^\star = \arg\min_k \frac1N\sum_{i=1}^N \Big(\tfrac16 \sum_{t\in H_m}\big(\hat y^{S}_{i,t}\,e^{\hat r_{i,t}(\theta_i^{(k)})} - y_{i,t}\big)^2\Big)^{1/2}. \]

Training stops after 5 rounds without improvement. Deng et al.'s convergence analysis is for stochastic gradient steps under their smoothness assumptions. It does not cover full-batch Adam on a GRU with round selection on a holdout. The paper is cited for the algorithm, not for a guarantee here.

## 5. Shrinkage and forecast

With \(\hat r_{i,t} = \hat r_{i,t}(\theta_i^{(k^\star)})\), choose for each state

\[ \lambda_i = \arg\min_{\lambda \in \{0, 0.5, 1\}} \Big(\tfrac16\sum_{t\in H_m}\big(\hat y^{S}_{i,t}\,e^{\lambda \hat r_{i,t}} - y_{i,t}\big)^2\Big)^{1/2}, \]

with ties broken toward the smaller \(\lambda\). The forecast is

\[ \boxed{\;\hat y_{i,m} = \hat y^{S}_{i,m}\,\exp\!\big(\lambda_i\,\hat r_{i,m}\big)\;} \]

## 6. Properties

**Proposition 1 (reduction to SARIMA).** If \(\lambda_i = 0\), then \(\hat y_{i,m} = \hat y^{S}_{i,m}\). If \(\lambda_i = 0\) for every state, v3's predictions, and therefore every metric, equal SARIMA's.

*Proof.* \(e^{0} = 1\). ∎

**Proposition 2 (error identity).** Let \(r_{i,m} = \log y_{i,m} - \log \hat y^{S}_{i,m}\) and \(d_{i,m} = \lambda_i \hat r_{i,m} - r_{i,m}\). Then

\[ \log \hat y_{i,m} - \log y_{i,m} = d_{i,m}, \qquad \hat y_{i,m} - y_{i,m} = y_{i,m}\big(e^{d_{i,m}} - 1\big), \]

and SARIMA's errors are the same expressions with \(d_{i,m}\) replaced by \(-r_{i,m}\).

So at a single month, v3 has a smaller absolute error in crore than SARIMA if and only if \(\lvert e^{d_{i,m}} - 1\rvert < \lvert e^{-r_{i,m}} - 1\rvert\). It has a smaller absolute log error if and only if \(\lvert \lambda_i \hat r_{i,m} - r_{i,m}\rvert < \lvert r_{i,m}\rvert\).

*Proof.* \(\log\hat y_{i,m} = \log\hat y^{S}_{i,m} + \lambda_i\hat r_{i,m}\) and \(\log y_{i,m} = \log \hat y^{S}_{i,m} + r_{i,m}\). Subtract the two to get the log error, then exponentiate: \(\hat y_{i,m} = y_{i,m}e^{d_{i,m}}\). ∎

*Remark.* The map \(d \mapsto \lvert e^{d}-1\rvert\) is not symmetric. An over-forecast by \(d\) costs more crore than an under-forecast by the same \(d\). So an improvement in log error does not always mean an improvement in crore. Equal-state RMSE in crore, the selection metric, also weights each state by its scale.

**Proposition 3 (optimal shrinkage, population form).** Fix the residual forecaster, and treat \((r, \hat r)\) as random with \(\mathrm{E}[\hat r^2] > 0\). Let \(Q(\lambda) = \mathrm{E}[(r - \lambda \hat r)^2]\), the expected squared log error. Then:

1. \(Q\) is minimised at \(\lambda^\ast = \mathrm{E}[r\hat r]/\mathrm{E}[\hat r^2]\), with \(Q(\lambda^\ast) = \mathrm{E}[r^2] - \mathrm{E}[r\hat r]^2/\mathrm{E}[\hat r^2] \le Q(0)\). Equality holds if and only if \(\mathrm{E}[r\hat r] = 0\).
2. Let \(\lambda^\dagger = \Pi_{[0,1]}(\lambda^\ast)\), the clipped optimum. Then \(Q(\lambda^\dagger) \le Q(0)\), with strict inequality if and only if \(\mathrm{E}[r\hat r] > 0\).

*Proof.*
- \(Q(\lambda) = \mathrm{E}[r^2] - 2\lambda\,\mathrm{E}[r\hat r] + \lambda^2\,\mathrm{E}[\hat r^2]\) is a convex quadratic. Setting \(Q'(\lambda) = 0\) gives \(\lambda^\ast\), and substituting it gives \(Q(\lambda^\ast)\). This proves (1).
- For (2), first take \(\mathrm{E}[r\hat r] \le 0\). Then \(\lambda^\ast \le 0\), so \(\lambda^\dagger = 0\) and the two sides are equal.
- Now take \(\mathrm{E}[r\hat r] > 0\). \(Q\) is strictly decreasing on \([0, \lambda^\ast]\). If \(\lambda^\ast \le 1\), then \(\lambda^\dagger = \lambda^\ast\) and \(Q(\lambda^\ast) < Q(0)\). If \(\lambda^\ast > 1\), then \(\lambda^\dagger = 1 \in (0, \lambda^\ast)\), so \(Q(1) < Q(0)\). ∎

*What Proposition 3 does not give.*
- It concerns the population loss, at a \(\lambda\) chosen with knowledge of \(\mathrm{E}[r\hat r]\).
- v3 instead picks \(\lambda_i\) from three values using six holdout months, and scores in crore. Proposition 2 shows that crore and log error can rank forecasts differently.
- Whether \(\mathrm{E}[r\hat r] > 0\) out of sample is exactly what the validation run tests. The pre-validation diagnostics in the spec make a positive answer plausible but small: Ljung–Box rejects in 4 of 32 states, and the median E-Way Bill correlation is +0.03 to +0.13.

**Proposition 4 (holdout fallback).** For every state, the chosen \(\lambda_i\) gives a holdout RMSE no larger than SARIMA's on \(H_m\), where SARIMA here means the residual-fit SARIMA.

*Proof.* \(\lambda = 0\) is in the candidate set and gives the residual-fit SARIMA predictions on \(H_m\). The minimum over the set is at most that value. ∎

*Remark.* Proposition 4 is a statement about the selection data only. With six months per state and three candidates, a state can pick \(\lambda_i > 0\) by chance. The out-of-sample effect is measured by comparing v3 with SARIMA on the validation origins (decision rule 1), and the effect of shrinkage by comparing v3 with B5 (\(\lambda_i = 1\)). The holdout SARIMA (parameters from \(\mathcal{D}_{<m-6}\)) and the forecast SARIMA (parameters from \(\mathcal{D}_{<m}\)) also differ. That is the cost of keeping the holdout out of sample (Lemma 1).

## 7. What the experiment must establish

| Claim | Evidence required (fixed before any run) |
|---|---|
| Residual correction beats SARIMA | Equal-state RMSE of v3 < B0 (SARIMA) on validation: seed mean and at least 2 of 3 seeds. Then the same on the confirmation months, after the GST-only amendment |
| Federation helps | v3 < B1 (local residual GRU), under the same rule |
| Personalization helps | v3 < B2 (FedAvg residual GRU) |
| A deep model is needed | v3 < B3 (per-state ridge regression) |
| E-Way Bills carry the signal | v3 < B4 (GST-only inputs) |
| Shrinkage helps | v3 < B5 (\(\lambda_i = 1\)) |

None of these claims follows from Sections 2–6. Each one is an empirical result to be reported whichever way it comes out.

## References

- Allassak, N., Trichni, S., and Omary, F. (2026). Federated hybrid ARIMAX-LSTM for collaborative fan fault prognostics: a cement plant case study. *The International Journal of Advanced Manufacturing Technology*. Preprint doi:10.21203/rs.3.rs-7033228/v1.
- Deng, Y., Kamani, M. M., and Mahdavi, M. (2020). Adaptive personalized federated learning. arXiv:2003.13461.
- Zhang, G. P. (2003). Time series forecasting using a hybrid ARIMA and neural network model. *Neurocomputing*, 50, 159–175.
- Box, G. E. P., Jenkins, G. M., Reinsel, G. C., and Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley. Source for SARIMA and Kalman-filter one-step prediction.
