# Literature review for FH-PFL (#21)

> Reviewed 29 October 2026. Each entry below was read from the paper's own text: method, experiments and conclusion, plus the appendix where it mattered. The comparison table is in [`fh_pfl_matrix.md`](fh_pfl_matrix.md) and the method being positioned is in [`../methods/fh_pfl.md`](../methods/fh_pfl.md). The PDFs are kept locally in `docs/literature/papers/`, which git ignores; most are arXiv copies whose licences do not allow redistribution.

## What the reading changes for FH-PFL

1. **Choosing \(\alpha_i\) on six holdout points is a known risk.**
   - Ditto chooses its λ per device on local validation data. For devices with fewer than five validation points, it falls back to a fixed λ.
   - FedL2P reports that purely local tuning ("L2P") meta-overfits each client's validation set and does worse than plain fine-tuning.

   FH-PFL's agreed design picks \(\alpha_i\) on six points per state per round. Ablation A3 (fixed \(\alpha = 0.5\)) is therefore the comparison that decides whether that choice is defensible.
2. **Local training often beats FedAvg in federated forecasting.**
   - CNFGNN reports GRU with FedAvg worse than local GRUs on both traffic datasets.
   - NeighborFL finds local-only training best for 10 of 26 sensors.

   This matches the thesis result that the local GRU beats FedAvg on validation, so it is not an anomaly of this panel.
3. **A fixed similarity graph from outside information can help.**
   - CNFGNN's GRU+FMTL baseline regularises each node toward its neighbours on a fixed Gaussian kernel of road distance. It beats both local GRUs and FedAvg.
   - MOCHA allows the task-relationship matrix to be known in advance.

   This supports testing FH-PFL's hypothesis: a fixed kernel from economic profiles is the same kind of object.
4. **Parameter similarity is fragile for neural networks.**
   - IFCA notes that two similar networks can have distant parameters because hidden units can be permuted.
   - pFedSim runs a FedAvg warm-up before measuring similarity for that reason.

   This is an argument for profile similarity. Any parameter-similarity ablation must start all states from the same seeded initialisation, which FH-PFL already does.
5. **Sample-count weighting is the default and sometimes helps.** FedAvg, FedPer, LG-FedAvg, IFCA and CFL weight by sample count, and pFedGraph's ablation favours it. FH-PFL's equal weighting is a deliberate departure; ablation A4 checks it.
6. **How large the private part should be is unsettled.** FedPer finds that one personal layer helps but sees no clear trend beyond that. FH-PFL's private head is a 17-parameter linear read-out, so ablation A2 tests very little capacity.

## Personalized federated learning

### FedPer — Arivazhagan, Aggarwal, Singh and Choudhary, 2019

*Federated Learning with Personalization Layers.* arXiv:1912.00818. The manuscript is marked "under review by AISTATS 2020"; no peer-reviewed version was found.

- **Method:** each client's network is base layers \(W_B\) plus personal layers \(W_{P_j}\). Clients run SGD on both. The server averages only \(W_B\), weighted by \(n_j\). With no personal layers it reduces to FedAvg.
- **Evidence:** CIFAR-10 and CIFAR-100 with 10 clients, each restricted to \(k\) classes, and FLICKR-AES image-aesthetics ratings from 30 users; ResNet-34 and MobileNet-v1. FedPer beats FedAvg most when data are least identical, and the gap closes as \(k\) approaches IID. Having at least one personal layer helps; the best number varies by dataset.
- **For FH-PFL:** this is the origin of the private-head idea. FH-PFL's head is FedPer with one personal layer and a GRU base.

### LG-FedAvg — Liang, Liu, Ziyin, Allen, Auerbach, Brent, Salakhutdinov and Morency

*Think Locally, Act Globally: Federated Learning with Local and Global Representations.* NeurIPS 2019 Workshop on Federated Learning; arXiv:2001.01523.

- **Method:** the inverse of FedPer. Each device keeps a local representation model and shares a smaller global model that operates on those representations. The global part is averaged by sample count.
- **Evidence:** MNIST, CIFAR-10, a 14-person mobile mood-prediction dataset, new rotated-MNIST devices and fair representations on UCI Adult. On mood prediction, a split between local and global layers beats both purely local and purely global models.
- **For FH-PFL:** it shows the split can go either way. FH-PFL shares the encoder and keeps the head, following FedPer and FedRep rather than LG-FedAvg.

### pFedMe — Dinh, Tran and Nguyen, NeurIPS 2020

*Personalized Federated Learning with Moreau Envelopes.* arXiv:2006.08848.

- **Method:** a bi-level problem. The global model \(w\) minimises the average of the clients' Moreau envelopes \(F_i(w) = \min_{\theta_i} f_i(\theta_i) + \frac{\lambda}{2}\lVert\theta_i - w\rVert^2\); each personal \(\theta_i\) is the proximal point. λ is excluded from the extremes 0 (no federation) and ∞ (no personalisation).
- **Evidence:** MNIST and synthetic data with multinomial logistic regression and a small DNN. With tuned hyperparameters the personal models beat FedAvg and Per-FedAvg.
- **For FH-PFL:** a proximal pull toward a shared model, like FH-PFL's \(\mu\) term, but toward a single global model rather than per-state aggregates.

### Ditto — Li, Hu, Beirami and Smith, ICML 2021

*Ditto: Fair and Robust Federated Learning Through Personalization.* PMLR 139, pp. 6357–6368; arXiv:2012.04221.

- **Method:** each device solves \(\min_{v_k} F_k(v_k) + \frac{\lambda}{2}\lVert v_k - w^*\rVert^2\), where \(w^*\) is the global FedAvg model. λ = 0 gives purely local models, and large λ gives the global model. The global and personal updates alternate in each round.
- **Evidence:**
  - Fashion-MNIST, FEMNIST and CelebA, under data and model poisoning.
  - Ditto is the most accurate and fair across devices in most settings.
  - It matches or beats APFL, MOCHA, L2SGD, EWC and Per-FedAvg on clean data.
- **λ is chosen per device on local validation data,** from a three-value grid. Devices with fewer than five validation points use a fixed λ instead (paper, Section 4 setup).
- **For FH-PFL:** the closest precedent for choosing the sharing weight per client on local validation. It also shows the authors did not trust that choice on tiny validation sets, which is FH-PFL's situation with six holdout months.

### FedBN — Li, Jiang, Zhang, Kamp and Dou, ICLR 2021

*FedBN: Federated Learning on Non-IID Features via Local Batch Normalization.* arXiv:2102.07623.

- **Method:** FedAvg, except that batch-norm parameters stay local and are never averaged. There are no extra hyperparameters. A two-layer network analysis (via the neural tangent kernel) shows faster convergence than FedAvg under feature shift.
- **Evidence:**
  - Five digit datasets as five clients.
  - Office-Caltech-10, DomainNet and four ABIDE medical sites.
  - Over 5 seeds, FedBN beats FedAvg and FedProx on the digit clients. The gain is largest on SVHN, whose appearance differs most, and the variance across runs is smaller.
- **For FH-PFL:** it supports keeping a small client-specific part when clients differ in scale or feature distribution. The GRU has no batch norm, so FedBN is background only. FedAP builds on it.

### IFCA — Ghosh, Chung, Yin and Ramchandran, NeurIPS 2020

*An Efficient Framework for Clustered Federated Learning.* arXiv:2006.04088.

- **Method:** \(k\) cluster models. Each client picks the model with the lowest local loss, \(\hat j = \arg\min_j F_i(\theta_j)\), and the server averages within clusters. A variant shares lower layers across clusters.
- **Evidence:** synthetic linear models, rotated MNIST and CIFAR, and FEMNIST. On FEMNIST IFCA reaches 88.0% against 84.5% for the global model and 75.9% for local models, and is robust to the choice of \(k\).
- **A caution (Section 4.1):** clustering on locally trained parameters fails for neural networks, because similar models can have distant parameters when hidden units are permuted.
- **For FH-PFL:** hard clustering is a coarse alternative to the soft profile kernel. The permutation point matters for any parameter-similarity comparison.

### CFL — Sattler, Müller and Samek

*Clustered Federated Learning: Model-Agnostic Distributed Multitask Optimization Under Privacy Constraints.* IEEE Transactions on Neural Networks and Learning Systems 32(8), pp. 3710–3722, 2021.

- **Method:** run FedAvg to a stationary point. If any client's gradient norm stays above a threshold, compute cosine similarities of client updates and split the clients into two groups, minimising the largest cross-group similarity. Recurse. The split is accepted only when a separation criterion holds.
- **Evidence:** CNNs on MNIST and CIFAR-10 split across 20 clients with label swaps, and a two-layer LSTM language model on AG News split by topic. CFL improves on FedAvg when the clients really do fall into clusters; on AG News, clustered clients reach lower perplexity than the federated model, which stalls at about 42.
- **For FH-PFL:** a principled way to decide that clients should not share. It is a hard split from gradients, not a soft weight from data profiles.

### MOCHA — Smith, Chiang, Sanjabi and Talwalkar, NIPS 2017

*Federated Multi-Task Learning.* arXiv:1705.10467.

- **Method:** \(\min_{W,\Omega} \sum_t \sum_i \ell_t(w_t^\top x_{it}, y_{it}) + \lambda_1 \mathrm{tr}(W\Omega W^\top) + \lambda_2\lVert W\rVert_F^2\). Ω models task relationships, either known in advance or learned. The algorithm is primal–dual and tolerates stragglers.
- **Evidence:** Google Glass, human-activity recognition and vehicle-sensor data with linear SVMs. Multi-task learning beats both local and global models on all three, for example 0.46% against 1.34% (local) and 2.23% (global) error on human activity.
- **Limitation, stated by the authors:** it applies only to convex models in its published form.
- **For FH-PFL:** the formal ancestor of a given client-relationship matrix. CNFGNN later used it with a road-distance graph as a forecasting baseline.

### FedALA — Zhang, Hua, Wang, Song, Xue, Ma and Guan, AAAI 2023

*FedALA: Adaptive Local Aggregation for Personalized Federated Learning.* arXiv:2212.01197.

- **Method:** before each round, the client initialises with \(\hat\Theta_i = \Theta_i^{t-1} + (\Theta^{t-1} - \Theta_i^{t-1}) \odot [\mathbf{1}; W_i^p]\). That is a learned element-wise mix of the global and local models, on the top \(p\) layers only. \(W_i^p\) is trained on an \(s\)% sample of local data and clipped to \([0, 1]\). After the first rounds it barely changes.
- **Evidence:** five vision and text benchmarks; it beats eleven baselines, including FedAMP, Ditto, FedRep and FedFomo. The module also improves other methods.
- **For FH-PFL:** the fine-grained version of FH-PFL's single \(\alpha_i\). It learns the mix on local training data rather than on a held-out split.

### FedL2P — Lee, Kim, Li, Qiu, Hospedales, Huszár and Lane, NeurIPS 2023

*FedL2P: Federated Learning to Personalize.* arXiv:2310.02420.

- **Method:** small meta-networks map each client's data statistics to personalisation hyperparameters: how much to use local versus global batch-norm statistics, and per-layer fine-tuning rates. The meta-networks are trained with federated averaging, using hypergradients on each client's validation loss.
- **Evidence:** CIFAR-10, CIFAR-10-C, Office-Caltech-10, DomainNet and Speech Commands with ResNet-18. FedL2P beats the hand-set fine-tuning baselines, and it generalises to clients it never saw.
- **Local-only variant:** "L2P" tunes each client alone and does worse than standard fine-tuning, because it meta-overfits each client's validation set (Section 4.2).
- **For FH-PFL:** client statistics are used here to set how much to personalise, not whom to share with. The L2P result is direct evidence on the risk of choosing \(\alpha_i\) per state.

### pFedSim — Tan, Zhou, Liu, Wang and Yu, 2023

*pFedSim: Similarity-Aware Model Aggregation Towards Personalized Federated Learning.* arXiv:2305.15706v1, marked "Preprint. Under review". No peer-reviewed version was found on 29 October 2026, so it must be cited as a preprint.

- **Method:** a FedAvg warm-up for half the rounds, then personalisation. Classifiers (heads) stay local. Each client's feature extractor is aggregated as \(\omega_i = \sum_j \Phi_{ij}\,\omega_j / \sum_j \Phi_{ij}\). The weights come from cosine similarity between the clients' classifier vectors, floored at zero and passed through a negative logarithm. Φ starts as the identity matrix.
- **Evidence:**
  - CIFAR-10, CINIC-10, EMNIST and Tiny-ImageNet, each split across 100 clients with Dirichlet label skew.
  - pFedSim beats FedRep, FedPer, FedFomo and others by 2.96 percentage points on average.
  - Its FedAP baseline is "f-FedAP", a variant warmed up with FedBN instead of a pre-trained model.
  - A motivating toy study with LeNet-5 on CIFAR-10 finds two things:
    - classifier layers separate client models better than earlier layers, by CKA similarity;
    - classifier similarity follows label overlap between clients.
- **For FH-PFL:** the same structure (a personalised encoder aggregate plus a private head) with a different similarity input. Its evidence concerns label skew in classification, which has no counterpart in one-step revenue regression.

### FedAP — Lu, Wang, Chen, Qin, Xu, Dimitriadis and Qin

*Personalized Federated Learning with Adaptive Batchnorm for Healthcare.* IEEE Transactions on Big Data, 2022 (doi:10.1109/TBDATA.2022.3177197); arXiv:2112.00734.

- **Method:**
  - Clients compute statistics of a pre-trained model's layer outputs on their data.
  - The server turns them into Wasserstein-type distances \(d_{ij}\) and computes a weight matrix **once**: \(w_{ii} = \lambda\) and \(w_{ij} = (1-\lambda)\,\hat w_{ij}\), with \(\hat w_{ij} \propto 1/d_{ij}\) over \(j \ne i\).
  - Clients keep their batch-norm layers local.
  - λ is one global hyperparameter, swept from 0.1 to 0.9.
- **Evidence:** five healthcare benchmarks. FedAP beats FedBN, and removing the weighting (which leaves FedBN) loses accuracy.
- **For FH-PFL:** the closest prior method. FH-PFL keeps its structure and changes the similarity input (economic profiles) and the self-weight (\(\alpha_i\) per state).

## Federated forecasting

### CNFGNN — Meng, Rambhatla and Liu, KDD 2021

*Cross-Node Federated Graph Neural Network for Spatio-Temporal Data Modeling.* doi:10.1145/3447548.3467371; arXiv:2106.05223.

- **Method:** each node runs a GRU encoder–decoder. The server runs a graph network over the nodes' hidden states along the road graph, trained by split learning alternating with FedAvg of the node models. Hidden states and gradients are sent, not only weights.
- **Evidence:**
  - PEMS-BAY and METR-LA traffic speed, 12-step forecasts.
  - RMSE on PEMS-BAY:

    | Model | RMSE |
    | --- | --- |
    | Local GRU | 4.010 |
    | GRU with FedAvg | 4.432–4.512 |
    | GRU+FMTL | 3.955–3.961 |
    | CNFGNN | 3.822 |

    GRU+FMTL is a MOCHA-style regulariser on a Gaussian-kernel road-distance graph.
  - On METR-LA, GRU with FedAvg (12.058) is again worse than the local GRU (11.801).
- **For FH-PFL:** FedAvg loses to local training on time series, and a fixed side-information kernel recovers the loss. The architecture itself (a server-side graph network sharing hidden states) is out of scope.

### NeighborFL — Chen, Meese, Nejad and Shen

*Individualized Federated Learning for Traffic Prediction with Error-Driven Aggregation.* ACM Journal on Autonomous Transportation Systems, 2026 (doi:10.1145/3796508); arXiv:2407.12226.

- **Method:**
  - Each traffic sensor aggregates its own model with a growing set of "favourite neighbours".
  - Candidates come from within a haversine radius and are tried nearest first.
  - A candidate is kept only if adding its model lowered the device's real-time prediction error.
  - A removal trigger drops neighbours either by a running reputation score or in reverse order of addition.
  - A Yao garbled-circuit option keeps device coordinates private.
- **Evidence:**
  - 26 PEMS-BAY sensors within a 1-mile radius, an online LSTM, 250 rounds and one random seed (40).
  - The best configuration lowers last-24-round mean squared error by 16.9% against naive FedAvg, from 8.97 to 7.45, with pretrained initial models.
  - "r-NaiveFL" aggregates each device with every neighbour inside the radius and skips the error test. It reaches 8.35 with pretraining. Without pretraining it reaches 10.88, which edges two of the four NeighborFL configurations. A fixed neighbourhood from side information gets most of the gain.
  - Local-only training ("Central") beat all three federated baselines for 10 of the 26 sensors.
- **For FH-PFL:** side information picks the candidates and validation error decides inclusion. That is the same split as FH-PFL's profile kernel plus holdout-chosen \(\alpha_i\). The single seed limits how far its numbers can be relied on.

### FedTime-Clust — Hung (Zenodo lists the author as Tien Hung Giang), 2025

*FedTime-Clust: A Novel Personalized Federated Learning Framework for Non-IID Time-Series Forecasting.* Zenodo, doi:10.5281/zenodo.17308847. Single author and self-deposited; the record names no venue, and no peer review was found.

- **Method, from the full text:**
  - K-means on the uploaded **model parameters** each round. The abstract's "similar time-series characteristics" is achieved through parameters, not data features.
  - Cluster count chosen by silhouette score.
  - Each client blends \(\tilde w_i = \alpha_i w_i + (1-\alpha_i) w_c\), with \(\alpha_i = \tanh(\lVert w_i - w_c\rVert^2)\).
- **Evidence:**
  - 100 clients simulated from one household's power series, split by time period or by weekday versus weekend.
  - LSTM, 50 rounds, compared only with FedAvg, FedProx, local and centralised training.
  - No seeds or spread are reported, the authors acknowledge the single-source simulation, and some references do not match their claimed content.
- **For FH-PFL:** shows that a local/cluster blend with a per-client weight is being tried in forecasting. It is too weak to cite as evidence for anything; cite it only as an example of the idea, if at all.

## Hierarchical coherence (novelty check for FH-PFL v2)

Searched on 30 October 2026 for "federated forecast reconciliation", "federated hierarchical forecasting", "privacy-preserving reconciliation" and "ADMM sharing forecasting". The FH-PFL v2 idea has five parts:
- private per-state models;
- only forecasts are exchanged;
- training is coupled by an aggregation constraint to a public total;
- the coupling is solved by ADMM;
- at the target month the forecasts are anchored to a forecast of the national total.

### Tan, Li, Yang, Zhao and Ju, 2021 — the closest match

*A Multi-Level Collaborative Load Forecasting Method for Distribution Networks Based on Distributed Optimization.* Journal of Shanghai Jiao Tong University 55(12), pp. 1544–1553 (doi:10.16183/j.cnki.jsjtu.2021.296). In Chinese with an English abstract. Read in full from the PDF page images.

- **Method:**
  - One LSTM per bottom-level region, with parameters θ_i.
  - The objective is \(\min \sum_i \lVert f_i(\theta_i; x_i) - y_i\rVert^2\), subject to equality constraints: each parent's actual load equals the sum of its children's model outputs, at every level up to the top.
  - The constraints enter an augmented Lagrangian with multipliers λ and penalty ρ. ADMM updates the models one after another, Gauss–Seidel style, then the multipliers.
  - Training runs over a peer-to-peer federated architecture in which the models exchange only their outputs.
  - Each LSTM is first pre-trained locally.
  - Prediction is bottom-up: the bottom-level forecasts are summed.
- **Evidence:**
  - ISO New England hourly load, 2013–2016, in a three-level hierarchy of four bottom regions, split 80:20.
  - LSTMs with 4 hidden layers of 100 units, 200 pre-training and 500 federated iterations, λ initialised at 0.1, ρ = 0.1.
  - Average MAPE falls from 2.36% for independent forecasts and 2.75% for traditional bottom-up to 1.33%.
  - One run, no seeds reported.
- **For FH-PFL v2:** this is the same mechanism: private models, an aggregation equality constraint during training, ADMM, and only outputs exchanged.
- **What v2 adds:**
  - a server that holds the public total and sees only a secure sum, instead of peer-to-peer exchange;
  - a Jacobi rather than Gauss–Seidel update;
  - at the target month, the anchor is a forecast of the public national series, where Tan et al. use bottom-up aggregation.

  These are variations, not a new method. v2 must be presented as an application and extension of Tan et al. (2021) to Indian state GST.

### Bai and Pinson, Energies 2019

*Distributed Reconciliation in Day-Ahead Wind Power Forecasting.* Energies 12(6):1112 (doi:10.3390/en12061112). Open access.

- **Method:**
  - Post-hoc reconciliation.
  - Constrained game-theoretic optimal projection (GTOP), written as an ADMM "sharing" problem: each wind farm adjusts its base forecast and shares only the adjustment Δy_k.
  - An aggregator enforces Σ ỹ_k = ỹ_AGG.
- **Evidence:** simulated, NREL and Sardinia wind data. Centralized MinT is best; the distributed ADMM version is close to its centralized counterpart.
- **For FH-PFL v2:** the ADMM sharing step at forecast time is this method. v2 differs only by also coupling during training.

### CoRe — Cristian, Harsha, Perakis and Quanz, 2025

*CoRe: Coherency Regularization for Hierarchical Time Series.* arXiv:2502.15983, a manuscript under submission.

- **Method:** one centralized network over all series, with a penalty \(\lVert W_L - AW_L\rVert + \lVert b_L - Ab_L\rVert\) on its final layer that bounds incoherence for any input.
- **Evidence:**
  - Labour, Traffic and Tourism data, 10 runs.
  - Coherence regularization improves out-of-sample error over an unregularized base model, most on the smaller datasets.
  - Too much weight on coherence hurts, with a best weight of 0.04 on Labour.
- **For FH-PFL v2:** independent evidence that a coherence penalty during training can help generalization on small data. Its tuning result supports choosing ρ on validation.

### Rangapuram, Werner, Benidis, Mercado, Gasthaus and Januschowski, ICML 2021

*End-to-End Learning of Coherent Probabilistic Forecasts for Hierarchical Time Series.* PMLR 139.

- **Method:** a centralized DeepVAR. Its samples are projected onto the coherent subspace with \(M = I - A^\top(AA^\top)^{-1}A\), and the loss is computed on the projected samples.
- **Evidence:** Labour, Traffic, Tourism, Tourism-L and Wiki, 5 runs. It has the best CRPS on most datasets against MinT, ERM and PERMBU-MinT.
- **For FH-PFL v2:** the centralized end-to-end counterpart. It needs all series in one model, so it is not federated.

## Statistical-plus-neural residual hybrids (novelty check for FH-PFL v3)

Searched on 30 October 2026 for "federated hybrid ARIMA LSTM residual", "federated residual correction of a statistical baseline" and "tax revenue hybrid SARIMA neural network residual". The v3 idea has four parts:
- a per-state statistical model;
- a neural network trained on its residuals;
- federated, personalized training of that network;
- per-state shrinkage of the correction.

### Allassak, Trichni and Omary — the closest match

*Federated hybrid ARIMAX-LSTM for collaborative fan fault prognostics: a cement plant case study.* The International Journal of Advanced Manufacturing Technology, published January 2026. Preprint: Research Square, doi:10.21203/rs.3.rs-7033228/v1. Read in full from the preprint. The journal version was not available.

- **Method:**
  - Each client fits an ARIMAX(2,1,2) on its own data with exogenous sensor inputs.
  - An LSTM is trained locally on that client's ARIMAX residuals.
  - Only the LSTM is federated: 5 local epochs, FedAvg in Flower, 10 rounds.
  - The forecast is the ARIMAX output plus the LSTM residual forecast.
- **Evidence:**
  - Three industrial fans, about 4,600 hourly samples each in 2023, with a chronological 80:20 split.
  - The hybrid beats standalone ARIMAX, LSTM and XGBoost.
  - The federated model is close to the centralized hybrid: R² 0.983 against 0.988.
  - One run, no seeds. Hyperparameters were grid-searched without a stated separate validation split.
- **For FH-PFL v3:** this is v3's ablation B2, a FedAvg residual network on a local statistical base.
- **What v3 changes:**
  - APFL personalization;
  - a scale-free log-ratio residual;
  - per-state shrinkage chosen on a holdout that the base model's parameters have not seen;
  - rolling origins.

  These are variations. v3 must be presented as an application and extension of this design to Indian state GST, not as a new method.

### Other entries

- **Federated hybrid carbon-emission forecasting (arXiv 2606.22618), method section checked:**
  - Fourteen country clients.
  - ARIMA trend and GARCH volatility outputs are used as *input features* to an LSTM-attention plus XGBoost model, trained with FedAvg.
  - The statistical models are feature generators. Nothing models their residuals.
- **FedRes, Federated Residual Learning (arXiv 2003.12880), abstract and introduction read:** the prediction is the sum of a shared global model and a personal local model, each learning against the other's residual. The residual idea is generic. There is no statistical base.
- **FedRMamba (doi:10.1145/3774904.3792712) and FedRSA (IEEE Access 2026, doi:10.1109/access.2026.3680223), abstracts only:**
  - FedRMamba is personalized FL with a shared low-frequency path plus local high-frequency residuals.
  - FedRSA injects client-specific seasonal residual signals into a shared backbone.
  - Both are neural decompositions, not a statistical model plus a neural correction.
- **Thayyib et al. (2023), already in the matrix:** results checked again. On national monthly Indian GST, ARIMA–ANN and ARIMA–NNAR residual hybrids did **not** beat Theta-TBATS. That is direct domain evidence against expecting a large gain from v3.

## Entries read earlier

FedAvg, FedProx, APFL and FedRep, together with FedAMP, FedFomo, pFedGraph and SFL, are summarised in the matrix. Their update rules were checked against the papers' equations or method sections on 29 October 2026.
