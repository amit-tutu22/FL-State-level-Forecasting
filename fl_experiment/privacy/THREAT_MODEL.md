"""Honest threat model for the FL State-level Forecasting research prototype.

This document describes what the system does and does **not** protect today.
It is intentionally conservative so README privacy language cannot oversell the code.
"""

# Threat model — Phase 5

## Assets

| Asset | Sensitivity | Notes |
|-------|-------------|-------|
| Raw fiscal / education / welfare tables on each client | High | Must not leave the client in this design |
| Model parameters / updates sent to the server | Medium | Can leak training information via inversion / membership attacks |
| Participation metadata (which states join which rounds) | Low–Medium | Reveals operational patterns |
| Global model | Medium | Shared artifact; may memorize rare local patterns |

## Actors

| Actor | Assumptions |
|-------|-------------|
| Honest-but-curious server | Follows FedAvg protocol but may inspect all updates and logs |
| Peer clients | Do not collude with the server in the baseline threat model |
| External network eavesdropper | Relevant only for multi-host deployments (not the local Ray simulation) |
| Malicious client | Out of scope for Phase 5 (no robust aggregation) |

## Trust boundaries

```
[State / sector client]
   local raw data  ──never sent──►
   model update    ──sent───────►  [Aggregation server]
                                      └──► global model
```

Simulation (`start_simulation`) runs clients as local Ray actors: there is **no** cross-machine network path and therefore **no TLS** in the default research workflow.

## Protections implemented (Phase 5)

1. **Data minimization by design** — only model updates leave clients (FedAvg).
2. **Client-side DP on updates** — after local training, the parameter *delta* is L2-clipped and Gaussian noise is added before upload (`dp_utils.py`, `run_phase5_privacy.py`).
3. **Privacy–utility reporting** — RMSE vs noise multiplier (+ rough ε estimate) is written to `results/phase5_privacy.csv`.

## Not implemented (do not claim)

| Mechanism | Status |
|-----------|--------|
| Secure Aggregation (SecAgg / SecAgg+) | Not implemented |
| TLS / mTLS for Flower server↔client | Not configured (simulation is local) |
| Formal (ε, δ)-DP accounting with Opacus / TF-Privacy | Approximate composition only |
| User-level / record-level DP on raw datasets | Not implemented |
| Robust aggregation / Byzantine defense | Not implemented |
| Homomorphic encryption of updates | Not implemented |

## What DP here means

This is **update-level Gaussian DP** (clip + noise on `w_local - w_global`), a common FL research baseline. It reduces (does not eliminate) leakage from inspecting updates. The ε numbers from `run_phase5_privacy.py` are **rough composition estimates** for communication, not a certification-grade privacy proof.

## Residual risks

- Strong adversaries with auxiliary data may still attack the global model.
- Very small noise multipliers give weak privacy and must not be marketed as “privacy-preserving deployment ready.”
- Welfare leakage labels and fiscal series are currently synthetic; real deployments need legal review (DPDP Act / sector policy), DPIA, and stronger crypto (SecAgg + TLS) before pilots.
