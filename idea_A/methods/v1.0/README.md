# Idea A v1.0

**Name:** Adaptive Identification under Endogenous Feedback Censoring.

**Allocation:** stratified Neyman allocation + exploration floor, then IPW / Hájek / AIPW using the **known** audit propensity `q`. This is classical two-phase sampling. It is **not** a novel algorithm.

## Assumptions

1. Logged context `X`, behavior action `π`, and natural observation bit `S` are available for every unit.
2. The factual reward `Y` is visible iff `O_t = S_t OR A_t`. The allocator and the estimators must not read hidden labels.
3. Audits `A_t ~ Bernoulli(q_t)` are independent of hidden `Y` given the logged history. `q_t` is known and written to the log.
4. Finite-population estimands are defined on the simulated world. Paired methods share that world.
5. If the natural channel depends on unobserved `Y` (MNAR) or has zero overlap, that channel does not identify the estimand. That is identification failure, not an estimator bug. An independent audit with `q_t > 0` is a different design.

## Estimands

| Estimand | Definition in this v1.0 |
| --- | --- |
| `population_mean_reward` | Finite-population mean of factual `Y`. |
| `preference_rate` | Finite-population `P(Y > 0)`. |
| `context_conditioned_reward` | Finite-population `E[Y \| stratum = h]` for each discrete stratum. |
| `policy_value_target_distribution` | `E_{x ~ p*}[Y]` with `p*` **uniform over strata**. Density ratio `w = p*(z)/p(z)`. This is a stated target **covariate** distribution, not a new action policy. |

## What v1.0 is not

Neyman allocation with an exploration mixture is in sampling textbooks. v1.0 exists to test whether a known audit channel recovers the estimands under endogenous censoring, and whether it beats uniform auditing at the same budget. Do not describe the allocator as a new method.
