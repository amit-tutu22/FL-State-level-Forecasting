# FH-PFL literature matrix (#21)

> **Status: draft.** Entries marked "checked" were compared against the paper's abstract and method description on 29 October 2026. Entries marked "to check" are from memory and must be read in full before they are cited in the thesis. The FH-PFL design is in [`docs/methods/fh_pfl.md`](../methods/fh_pfl.md).

FH-PFL has four parts:

1. a private prediction head;
2. per-state aggregation weights;
3. weights computed from training-period revenue profiles rather than from model parameters or losses;
4. a blend of the state's own encoder with its aggregate.

Each part already exists in published work. The table records which work owns which part.

| Work | Shared | How each client's aggregate is weighted | Personal part | Overlap with FH-PFL | Status |
|---|---|---|---|---|---|
| FedAvg — McMahan et al., AISTATS 2017 | Whole model | One global average weighted by sample count | None | Baseline | Implemented here |
| FedProx — Li et al., MLSys 2020 | Whole model | Global average | None; proximal term toward the global model | Proximal term in the local step | Implemented here |
| APFL — Deng et al., 2020 (arXiv 2003.13461) | Whole model | Global average | Local model mixed with the global one by a learned \(\alpha_i\) | Blend of own and shared parameters | Implemented here |
| FedPer — Arivazhagan et al., 2019 (arXiv 1912.00818) | Base layers | Global average | Personal top layers | Private head | To check |
| LG-FedAvg — Liang et al., 2020 (arXiv 2001.01523) | Top layers | Global average | Local representation layers | Inverse split (private encoder, shared head) | To check |
| FedRep — Collins et al., ICML 2021 | Representation (encoder) | Global average | Local head, trained first each round with the representation fixed | Private head with a shared encoder; FedRep averages uniformly across clients | Checked |
| FedBN — Li et al., ICLR 2021 | All but batch norm | Global average | Batch-norm layers | Partial sharing; no batch norm in this GRU | To check |
| Ditto — Li et al., ICML 2021 | Whole model | Global average | Separate personal model regularised toward the global one | Proximal coupling to a shared model | To check |
| pFedMe — Dinh et al., NeurIPS 2020 | Whole model | Global average | Moreau-envelope personal model | Proximal coupling | To check |
| FedAMP — Huang et al., AAAI 2021 | Whole model | Per-client "cloud model": a convex combination of all clients' models, with weights from **model-parameter similarity** (attentive message passing) | Personal model regularised toward its cloud model | **Closest mechanism.** Per-client similarity-weighted aggregation plus a proximal pull toward it. FH-PFL differs in the similarity source (data profiles fixed per origin) and in sharing only the encoder | Checked |
| FedFomo — Zhang et al., ICLR 2021 | Whole model | Per-client weights from each peer model's **loss improvement on the client's validation data** divided by parameter distance | Each client's own combination | Per-client weighted aggregation; FH-PFL needs no peer-model evaluation | Checked |
| pFedGraph — Ye et al., ICML 2023 | Whole model | Per-client collaboration graph from **pairwise model similarity and dataset size**, solved as a quadratic program each round | Local model regularised toward its aggregate | Per-client graph aggregation plus a proximal pull. FH-PFL fixes the graph from profiles and drops sample-size weighting | Checked |
| SFL — Chen et al., IJCAI 2022 ("Personalized Federated Learning With a Graph") | Whole model | GCN aggregation along a **predefined client-relation graph** (or a learned one) | Personalised model per client | **Closest in spirit.** A given graph from side information decides who shares with whom. FH-PFL's profile kernel is such a graph built from revenue statistics | Checked |
| FedALA — Zhang et al., AAAI 2023 | Whole model | Global model, adaptively and element-wise blended into each local model | Element-wise blend weights | Learned local/global blend, finer-grained than FH-PFL's single \(\alpha\) | To check |
| IFCA — Ghosh et al., NeurIPS 2020 | One model per cluster | Hard cluster assignment by lowest loss | Cluster membership | Grouping similar clients, but hard rather than soft | To check |
| CFL — Sattler et al., IEEE TNNLS 2021 | One model per cluster | Recursive bipartition by gradient cosine similarity | Cluster membership | Similarity-based grouping, from gradients | To check |
| MOCHA — Smith et al., NeurIPS 2017 | Task relationships | Multi-task learning with a learned task-relationship matrix | Per-client model | Learned client relations | To check |

## Forecasting applications

| Work | Setting | Relevance | Status |
|---|---|---|---|
| CPFL — natural gas load forecasting (FL-IJCAI workshop 2022) | Retailers' consumers clustered by a knowledge-based federated clustering, then attention-based aggregation within clusters | Similarity of load patterns, not model parameters, decides grouping; time-series forecasting | Checked (abstract) |
| Clustered federated load prediction with Louvain (Algorithms 2025, doi:10.3390/a18010032) | Clients clustered by cosine similarity of model updates; DP LSTM | Similarity-grouped FL for forecasting | Checked (abstract) |
| Client clustering versus personalization in residential load forecasting (Energy Reports 2026, doi:10.1016/j.egyr.2026.109681) | Benchmark: after per-household fine-tuning, clustered methods land within 2% of the global model | A warning that grouping gains can vanish against a strong personalised baseline. Here the strong baselines are SARIMA and the local GRU | Checked (abstract) |
| Regional economic forecasting with FL (doi:10.1145/3770177.3770335) | LSTM FL across Chinese provinces, quarterly GDP | Federated sub-national economic forecasting; no personalization | Checked (abstract) |
| Spatial models for cantonal tax revenue (KOF Analysen 2017) | Swiss cantonal tax forecasts with spatial econometrics | Non-federated precedent for sharing across regions in revenue forecasting | Checked (abstract) |

A search on 29 October 2026 found no paper combining personalized federated learning with sub-national tax-revenue forecasting. That is a search result, not proof of absence.

## What FH-PFL can and cannot claim

- **Not new:** private heads (FedPer, FedRep); per-client similarity-weighted aggregation (FedAMP, FedFomo, pFedGraph); a predefined client graph (SFL); proximal pull toward a personalised aggregate (FedAMP, pFedGraph, Ditto); local/shared blending (APFL, FedALA).
- **The narrow distinction:** the aggregation weights come from a fixed, leakage-checked profile of each state's training-period revenue dynamics (growth, volatility, seasonality, without level). They are recomputed at each rolling origin, applied to the encoder only, and tested against a uniform-weight ablation under a frozen chronological protocol on Indian state GST.
- **Wording for the thesis:** "a profile-weighted variant of personalized federated learning, tested as a hypothesis about which states should share", not "a novel algorithm". If A1 matches FH-PFL, that negative result is the finding.

## To do before citing

- Read FedPer, LG-FedAvg, FedALA, Ditto, pFedMe, IFCA, CFL and MOCHA in full and fill the remaining "to check" cells.
- Search again for "federated" with "tax", "GST", "revenue" or "fiscal" in economics venues, and for side-information or metadata-weighted aggregation (for example graph FL on traffic sensors with road-distance graphs).
- Confirm the FedAMP and pFedGraph update rules against their equations, not only their abstracts.
