# UAI 2026 implementation notes (**NOT CCF-A Main**)

Short appendix owned by this package. Does not replace `reports/v0/BASELINE_AUDIT.md`.

## Labels

| Method | Status |
| --- | --- |
| e-LOND (UAI γ_t=1/(t(t+1))) | `independent_reimplementation` |
| Closed e-LOND (Theorem 2 DP) | `independent_reimplementation` |
| Donation e-LOND (Eq. 8) | `independent_reimplementation` |
| r-LOND / Closed r-LOND / Donation r-LOND | `independent_reimplementation` |
| e-LORD (P1-B) | reused from SCORE `independent_reimplementation` |
| AISTATS 2024 LOND via evalue-omt | `aistats2024_official_wrapper` |
| offline closedeBH / closedBY | `offline_eclosure_official_clone_arxiv2509.02517` — **not** UAI online |

## Official clones (do not edit)

- `official/` SHA `8bbd5a37b48abb6d79f5e1eef95d4fe6315339b5` — offline e-Closure (arXiv:2509.02517). Python API `closedeBH`, `closedBY` needs C++ `_cebh`.
- `evalue-omt/` SHA `cda6e9962514924044143920cf1cadbec8738d52` — AISTATS 2024. Experiment name **`lag_comp`**.

## `lag_comp` mapping

`lag_comp` is the AISTATS 2024 local-dependence *lag grid* experiment (bounded sequential samples, δ=0.3, hypotheses=1000, max_seq_len=200). UAI 2026 Section 4 uses a different protocol: one Gaussian observation per hypothesis, m=200, L=100, Σ_ij=0.5^{|i-j|} on the lag band, δ=0.1. P0 uses the UAI DGP. The name `lag_comp` is recorded as the related official experiment, not as a drop-in.

## Metrics

FDR = E[FDP_T]. SupFDR = E[sup_t FDP_t]. Implemented as separate functions; never interchangeable.
