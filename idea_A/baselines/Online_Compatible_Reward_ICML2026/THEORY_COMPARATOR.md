# Drago et al. ICML 2026 — theory-only comparator

**Status: blocked / theory-only.** No official implementation was found. No author code was invented.

## Paper

- **Title:** Online Compatible Reward Identification from Preference Feedback
- **Authors:** Simone Drago, Marco Mussi, Alberto Maria Metelli (Politecnico di Milano)
- **Venue:** ICML 2026 (Proceedings of the 43rd International Conference on Machine Learning), PMLR 306:26365–26389
- **PMLR:** https://proceedings.mlr.press/v306/drago26a.html
- **PDF (PMLR GitHub assets):** https://raw.githubusercontent.com/mlresearch/v306/main/assets/drago26a/drago26a.pdf
- **Author PDF:** https://marcomussi.github.io/papers/icmlcompreward2026/paper.pdf
- **Poster:** https://icml.cc/virtual/2026/poster/65224
- **BibTeX key:** `pmlr-v306-drago26a`

## Code search (2026-10-30)

A brief search found **no official repository** for this paper:

| Source | Result |
|---|---|
| Web search for “Online Compatible Reward Identification” + github | Paper/PDF/poster only; no code link |
| GitHub repo search `compatible reward identification drago` | 0 repositories |
| https://github.com/simone-drago | Only `simone-drago.github.io` |
| https://github.com/marcomussi | 21 repos; closest is `SleepingRL` which is the **ICML 2025 Sleeping Reinforcement Learning** paper, not this 2026 work |
| PMLR page / ICML poster page | No code badge or GitHub URL |
| paperswithcode-style lookup | No implementation listing located |

This matches the prior finding that the paper is a theory comparator only.

## What the paper contributes (theory)

The paper studies **compatible reward identification** from preference queries, not policy optimization:

- Goal: recover a reward in the compatible set \(R(P,g)\) that is accurate on the **entire** trajectory/state-action space, not only on the occupancy of a learned policy.
- Deterministic preferences (\(g = H\)): lower bounds on the number of queries needed for a sufficient query set in offline and online settings, plus an inapproximability result (too few queries can yield arbitrarily large suboptimality).
- Stochastic Bradley–Terry preferences: query **basis** and its **index** as complexity measures; an \((\varepsilon,\delta)\)-PAC algorithm that identifies a nearly-optimal query basis with polynomial human-query complexity.

The authors explicitly contrast this with standard RLHF, which only needs accuracy on visited trajectories.

## Why this is not an executable baseline here

- No official code, configs, or synthetic scripts were released with the paper.
- Inventing an “author implementation” would not be a reproduction.
- The objects (query bases, compatible reward sets, PAC identification) are **not the same target** as Idea A’s finite-budget Neyman audit of an already-defined estimand.

## Comparability

`NOT DIRECTLY COMPARABLE` as an empirical competitor. Use only as a **theory-level** contrast: identification of a globally compatible reward vs. estimation of a target functional under selective labels.
