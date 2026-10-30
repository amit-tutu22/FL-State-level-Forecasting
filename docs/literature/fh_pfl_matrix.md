# FH-PFL literature matrix (#21)

> **Status: draft, reviewed 29 October 2026.** The status column uses three labels:
>
> - "Full text read" means the method, experiments and conclusion were read. Those papers have entries in [`review.md`](review.md).
> - "Method checked" means the update rule was compared against the paper's own text.
> - "Abstract only" entries must be read in full before they are cited for anything beyond what the abstract says.
>
> The FH-PFL design is in [`docs/methods/fh_pfl.md`](../methods/fh_pfl.md).

FH-PFL has five parts:

1. a private prediction head;
2. per-state aggregation weights;
3. weights computed from training-period revenue profiles, rather than from model parameters or losses;
4. a blend of the state's own encoder with its aggregate;
5. a per-state, per-round blend weight chosen on the inner holdout.

Every part already exists in published work. The closest single method is FedAP. It computes a fixed similarity matrix once from data statistics, gives each client a self-weight λ plus \((1-\lambda)\) times similarity-normalised weights on the others, and keeps some layers private. HeurFedAMP and pFedSim are close in the same way, but take similarity from model parameters.

## Personalized federated learning

| Work | Shared | How each client's aggregate is weighted | Personal part | Overlap with FH-PFL | Status |
|---|---|---|---|---|---|
| FedAvg — McMahan et al., AISTATS 2017 | Whole model | One global average weighted by sample count | None | Baseline | Implemented here |
| FedProx — Li et al., MLSys 2020 | Whole model | Global average | None; proximal term toward the global model | Proximal term in the local step | Implemented here |
| APFL — Deng et al., 2020 (arXiv 2003.13461) | Whole model | Global average | Local model mixed with the global one by a learned \(\alpha_i\) | Blend of own and shared parameters | Implemented here |
| FedPer — Arivazhagan et al., 2019 (arXiv 1912.00818; preprint, marked under review at AISTATS 2020) | Base layers | Global average weighted by sample count | Personalization layers on top, never sent | Private head on a shared base | Full text read |
| LG-FedAvg — Liang et al., NeurIPS 2019 FL workshop (arXiv 2001.01523) | Global model on top of local representations | Global average weighted by sample count | Local representation layers per device | The inverse split: private encoder, shared top | Full text read |
| FedRep — Collins et al., ICML 2021 | Representation | Global average | Local head, updated first each round with the representation fixed | Private head with a shared encoder; uniform averaging | Method checked |
| FedBN — Li et al., ICLR 2021 | All but batch norm | Global average | Batch-norm layers | Partial sharing; this GRU has no batch norm | Full text read |
| Ditto — Li et al., ICML 2021, PMLR 139:6357–6368 (arXiv 2012.04221) | Global model | Global average | Personal \(v_k\) minimising \(F_k(v_k) + \frac{\lambda}{2}\lVert v_k - w^*\rVert^2\); \(\lambda = 0\) is local-only, \(\lambda \to \infty\) is global | Proximal coupling to a shared model. λ is chosen **per device on local validation data**, with a fixed λ for devices that have fewer than five validation points; this bears directly on FH-PFL's \(\alpha_i\) chosen on six holdout months | Full text read |
| pFedMe — Dinh et al., NeurIPS 2020 | Global model | Global average | Personal model via a Moreau-envelope inner problem | Proximal coupling | Full text read |
| FedAMP — Huang et al., AAAI 2021 | Whole model | Per-client cloud model \(u_i = \sum_j \xi_{ij} w_j\). In FedAMP, \(\xi_{ij} = \alpha_k A'(\lVert w_i - w_j\rVert^2)\) with \(A(x) = 1 - e^{-x/\sigma}\). In HeurFedAMP, \(\xi_{ii}\) is a fixed self-weight hyperparameter and \(\xi_{ij} = (1-\xi_{ii})\,\mathrm{softmax}_{j \ne i}(\sigma \cos(w_i, w_j))\) | Personal model \(\arg\min_w F_i(w) + \frac{\lambda}{2\alpha_k}\lVert w - u_i\rVert^2\) | **Very close mechanism.** HeurFedAMP's self-weight plus softmax over the others is the same form as FH-PFL's blend. FedAMP's local step is the same proximal pull. Differences: similarity from model parameters, recomputed each round; the whole model is shared | Method checked |
| FedFomo — Zhang et al., ICLR 2021 | Whole model | Per-client weights from each peer model's loss improvement on the client's validation data, divided by parameter distance | Each client's own combination | Per-client weighting chosen on client validation data, as FH-PFL's \(\alpha_i\) is | Method checked |
| pFedGraph — Ye et al., ICML 2023 | Whole model | Per-client graph from a quadratic program that trades off cosine model similarity and relative dataset size | Local model trained with \(-\lambda \cos(\theta_i, \bar\theta_i)\), a cosine pull toward the aggregate | Per-client graph aggregation plus a pull toward it. Its ablation finds that using dataset size beats equal weights (its Tables 8–9), which bears on FH-PFL's unweighted choice and ablation A4 | Method checked |
| SFL — Chen et al., IJCAI 2022 | Whole model | GCN aggregation along a predefined client-relation graph, or a learned one | Personalised model per client | A given graph from side information decides who shares with whom | Method checked |
| FedAP — Lu et al., IEEE Trans. Big Data 2022 (arXiv 2112.00734) | All but batch norm | Weight matrix computed **once** from **data statistics** (batch-norm statistics of a pre-trained model's layer outputs, Wasserstein-style distance): \(w_{ii} = \lambda\), \(w_{ij} = (1-\lambda)\,\hat w_{ij}\) with \(\hat w_{ij} \propto 1/d_{ij}\) over \(j \ne i\). λ is one global hyperparameter, with a sensitivity sweep from 0.1 to 0.9 | Local batch-norm layers | **Closest overall.** A fixed, data-derived similarity matrix, a self-weight blend and private layers. Its ablation drops the weighting, which turns it into FedBN, and accuracy falls; this is the same kind of test as FH-PFL's A1. Differences: FH-PFL uses hand-chosen economic time-series statistics instead of network-feature statistics, a softmax kernel that includes self, \(\alpha_i\) chosen per state and round, and profiles recomputed at each rolling origin | Full text read |
| pFedSim — Tan et al., 2023 (arXiv 2305.15706; **preprint, not peer reviewed** as of 29 October 2026) | Feature extractor, aggregated per client over similar clients, after a FedAvg warm-up for half the rounds | \(\omega_i = \sum_j \Phi_{ij}\omega_j / \sum_j \Phi_{ij}\), with \(\Phi_{ij}\) a negative-log transform of the clipped cosine similarity between the two clients' classifier vectors | Classifier trained locally | **Same structure as FH-PFL:** a personalised encoder aggregate and a private head. The difference is the similarity source, heads versus profiles. Its evidence is label skew in classification | Full text read |
| FedALA — Zhang et al., AAAI 2023 | Whole model | Global model | Element-wise aggregation weights on the higher layers, learned on an s% sample of local data before each round | Learned local/global blend, chosen on local data; finer-grained than FH-PFL's single \(\alpha_i\) | Full text read |
| FedL2P — Lee et al., NeurIPS 2023 (arXiv 2310.02420) | Global model | Global average | Meta-nets, trained federatedly on validation hypergradients, map client statistics to how much to mix local and global batch-norm statistics and to per-layer fine-tuning rates | Client statistics set how much to personalise. Its local-only variant meta-overfits each client's validation set, which is a warning for per-state \(\alpha_i\) | Full text read |
| IFCA — Ghosh et al., NeurIPS 2020 | One model per cluster | Hard assignment to the cluster model with the lowest local loss | Cluster membership | Grouping similar clients, but hard rather than soft. Notes that clustering raw neural-network parameters is unreliable because hidden units can be permuted | Full text read |
| CFL — Sattler et al., IEEE TNNLS 32(8):3710–3722, 2021 | One model per cluster | Recursive bipartition by cosine similarity of client updates, after FL converges | Cluster membership | Similarity-based grouping, from gradients | Full text read |
| MOCHA — Smith et al., NIPS 2017 | Task relationships | Multi-task objective with a matrix \(\Omega\) of task relationships, either known a priori or learned | Per-client model | A client-relationship matrix that can be given in advance. Convex models only | Full text read |

## Federated forecasting with similarity between clients

| Work | Setting | Relevance | Status |
|---|---|---|---|
| FedASTA — 2024 (arXiv 2405.13090) | Traffic flow. Clients decompose series into seasonal and trend parts and upload trend features. The server builds an adaptive client graph from them, alongside a static distance graph | A client graph from time-series features rather than model weights, as FH-PFL's profiles are. It shares features, not encoder weights | Method checked (introduction) |
| NeighborFL — Chen et al., ACM J. Autonomous Transportation Systems 2026 (doi:10.1145/3796508; arXiv 2407.12226) | Traffic, 26 PEMS-BAY sensors, online LSTM, one seed. Each sensor aggregates with neighbours inside a 1-mile radius, kept only if they lower its real-time error | Side information picks candidates and error decides inclusion. Local-only training was best for 10 of 26 sensors. The fixed-radius baseline with no error test gets most of the gain | Full text read |
| CNFGNN — Meng et al., KDD 2021 | Traffic. GRU encoder–decoders on nodes, FedAvg, plus a server-side graph network on hidden states trained by split learning | GRU with FedAvg is worse than local GRUs on PEMS-BAY and METR-LA, while GRU+FMTL with a fixed Gaussian-kernel road-distance graph beats both | Full text read |
| FedTime-Clust — Hung, 2025 (Zenodo 10.5281/zenodo.17308847; self-deposited, not peer reviewed) | Household power, 100 clients simulated from one UCI series. K-means on **model parameters** (the abstract says time-series characteristics); blend \(\alpha_i = \tanh(\lVert w_i - w_c\rVert^2)\) | A local/cluster blend in federated forecasting. Weak evidence: one source series, no seeds, only FedAvg and FedProx as baselines | Full text read |
| CPFL — FL-IJCAI workshop 2022 | Natural gas load. Knowledge-based federated clustering, then attention-based aggregation in clusters | Load-pattern similarity decides grouping | Abstract only |
| Clustered DP load prediction with Louvain — Algorithms 2025 (doi:10.3390/a18010032) | Clients clustered by cosine similarity of updates; DP LSTM | Similarity-grouped federated forecasting | Abstract only |
| Clustering versus personalization in residential load forecasting — Energy Reports 2026 (doi:10.1016/j.egyr.2026.109681) | After per-household fine-tuning, clustered methods land within 2% of the global model | A warning that grouping gains can vanish against a strong personalised baseline. Here those baselines are SARIMA and the local GRU | Abstract only |

## Hierarchical coherence (prior work for FH-PFL v2)

| Work | Setting | Relevance | Status |
|---|---|---|---|
| Tan et al., J. Shanghai Jiao Tong Univ. 55(12), 2021 (doi:10.16183/j.cnki.jsjtu.2021.296) | Private LSTM per bottom-level load region. Training is constrained so the outputs add up to the parent actuals, solved by ADMM over peer-to-peer FL that exchanges only outputs. Predictions are summed bottom-up | **Same mechanism as FH-PFL v2.** v2 is an application and extension of it, not a new algorithm | Full text read |
| Bai and Pinson, Energies 2019 (doi:10.3390/en12061112) | Post-hoc reconciliation as an ADMM sharing problem; farms share only adjustments | The forecast-time ADMM step of v2 | Full text read |
| CoRe — Cristian et al., 2025 (arXiv 2502.15983) | Centralized network with a final-layer coherence regularizer | A coherence penalty during training helps on small data; too much hurts | Full text read |
| Rangapuram et al., ICML 2021 | Centralized DeepVAR with projection onto the coherent subspace inside training | End-to-end coherent training, not federated | Method checked |

## Statistical-plus-neural residual hybrids (prior work for FH-PFL v3)

| Work | Setting | Relevance | Status |
|---|---|---|---|
| Zhang, Neurocomputing 50, 2003 | ARIMA, then a neural network on its residuals, centralized | The classical hybrid that v3 builds on | Known reference |
| Allassak, Trichni and Omary, Int. J. Adv. Manuf. Technol. 2026 (preprint doi:10.21203/rs.3.rs-7033228/v1) | Local ARIMAX per client; LSTM on its residuals federated with FedAvg in Flower; 3 clients | **Same mechanism as v3's FedAvg ablation (B2).** v3 adds APFL, log-ratio residuals, per-state shrinkage and rolling origins. It is an extension, not a new method | Full text read (preprint) |
| Federated hybrid carbon-emission forecasting (arXiv 2606.22618) | ARIMA and GARCH outputs as input features to a FedAvg LSTM plus XGBoost | Statistical models used as features, not as a base with residual correction | Method checked |
| FedRes (arXiv 2003.12880) | Global plus local model, each fitted to the other's residual | Generic residual split, with no statistical base | Abstract and introduction |
| FedRMamba (doi:10.1145/3774904.3792712); FedRSA (doi:10.1109/access.2026.3680223) | Personalized FL with shared and local, or seasonal-residual, neural paths | Neural decompositions only | Abstract only |

## Revenue and economic forecasting

| Work | Setting | Relevance | Status |
|---|---|---|---|
| Thayyib et al., Cogent Economics & Finance 2023 (doi:10.1080/23322039.2023.2285649) | National monthly Indian GST, 2017–2022. Theta-TBATS hybrids beat ANN and NNAR | Indian GST precedent. Classical models beat neural ones, consistent with SARIMA leading here. The authors could not test state-level series because of data discrepancies | Results checked |
| Local tax revenue with GRU — Korea, 2025 (doi:10.63851/001c.140949) | Annual local tax by metropolitan and municipal government, against budget estimates during COVID-19 | GRU for sub-national revenue; not federated | Abstract only |
| Regional economic forecasting with FL (doi:10.1145/3770177.3770335) | LSTM FL across Chinese provinces, quarterly GDP | Federated sub-national economic forecasting; no personalization | Abstract only |
| Spatial models for cantonal tax revenue — KOF Analysen 2017 | Swiss cantonal tax forecasts with spatial econometrics | Non-federated precedent for sharing across regions in revenue forecasting | Abstract only |

Searches on 29 October 2026 found no paper that combines personalized federated learning with sub-national tax-revenue forecasting. They covered "federated" with "GST", "tax revenue", "fiscal revenue" and "municipal" in machine-learning and economics results. That is a search result, not proof of absence.

## What FH-PFL can and cannot claim

- **Not new:**
  - private heads or private layers (FedPer, FedRep, FedBN, FedAP, pFedSim);
  - per-client similarity-weighted aggregation (FedAMP, FedFomo, pFedGraph, pFedSim);
  - a fixed similarity matrix computed from data statistics (FedAP);
  - a predefined client graph (SFL, MOCHA with a known \(\Omega\));
  - a self-weight plus a normalised weight on the others (HeurFedAMP, FedAP);
  - a proximal pull toward a personalised aggregate (FedAMP, pFedGraph, Ditto);
  - blend weights learned on local data (APFL, FedALA, FedFomo);
  - a personalisation strength chosen per client on its own validation data (Ditto's per-device λ);
  - client graphs from time-series trend features (FedASTA).
- **What remains specific:**
  - the similarity input is an interpretable economic profile of each state's training-period revenue (growth, volatility and seasonal strength, without level);
  - it is recomputed at each rolling origin from fit months only;
  - it is applied to a GRU encoder, with the blend weight chosen per state on the inner holdout;
  - it is tested against a uniform-weight ablation under a frozen chronological protocol on Indian state GST.

  This is a FedAP- or HeurFedAMP-style aggregation with a domain-specific similarity input. It is not a new aggregation rule.
- **Wording for the thesis:** "a profile-weighted personalized FL variant, closest to FedAP and HeurFedAMP, used to test whether states with similar revenue dynamics should share more". Do not call it "a novel algorithm". If uniform weighting (A1) matches it, that negative result is the finding.
- **Comparisons this suggests:** model-parameter similarity, as in HeurFedAMP and pFedSim, is the natural alternative to profile similarity. An ablation that replaces \(z_i\) with HeurFedAMP's cosine similarity of encoder parameters would separate "similar states" from "similar models". It is not in the agreed ablation set and would need agreement before it is added.

## Remaining reading

- The rows still marked "Abstract only" (FedASTA beyond its introduction, CPFL, the Louvain and residential-load papers, and the revenue papers other than Thayyib et al.) must be read in full before they are cited beyond their abstracts.
