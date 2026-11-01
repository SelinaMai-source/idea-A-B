# SCORE implementation notes (ICML 2026 Main)

- **Paper:** Kuang, Gang, Xia — SCORE (PMLR v306 / arXiv:2601.20386)
- **Code status:** `independent_reimplementation` (no official repository)
- **Wealth:** full overshoot refund `C = (C - O)_+` with `O = (α e - 1)_+`, not “reject then add α”
- **SCORE-LOND:** Prop. 4.1, refund `min(O, α)` into the additive wealth term
- **SCORE-LORD / e-LORD:** Prop. 4.3 / §4.2, `ω_t ≡ 0.05` in the §6.1 default
- **SCORE-SAFFRON:** Prop. 4.4, candidate tightening then overshoot; `λ_t ≡ 0.5`
- **SCORE+:** Prop. 5.3 / Thm. 5.4 retroactive denominator `R_{t-1} ∨ 1`. Handmade stepwise wealth tests in `test_score_wealth.py` match the paper update; the v0 30-seed ~5× power gap vs e-LORD is **not** a detected formula bug, but is still unverified at paper-scale 500. Callable only with `assume_cpqd=True`; otherwise `status='N/A'` and `protocol_status='INVALID_PROTOCOL'`, and the procedure must not claim an FDR guarantee.
- **γ_t:** default `1/(t(t+1))` (infinite-horizon, not renormalized by T)
- **Power:** AP(t) = mean fraction of arrived non-nulls rejected by t; power ratio vs family baseline, pointwise in t
- **DGP §6.1:** independent Gaussian mixture, T=1000, e-value `φ(x;3,6)/φ(x;0,1)`. Paper-scale 500 reps is in config; runners refuse it without `--paper-scale`
