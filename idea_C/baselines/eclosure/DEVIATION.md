# Deviations (UAI 2026 wrappers / independent Python)

## C++ / Mac

Import of official `eclosure._cebh` failed in the v0 `.venv`:

```
The eclosure C++ bindings could not be imported.
Run `pip install .` from the repository root to build the extension.
```

Compile was **not** attempted (task: if Mac cannot use C++, skip compile and use Python). Offline `closedeBH` / `closedBY` are therefore **unavailable** at runtime; `try_offline_closedeBH` returns `status='unavailable'` with this reason. This does **not** block UAI online methods, which never needed that C++ path.

## AISTATS 2024 LOND wrapper

`evalue-omt` `alg/__init__.py` imports LORD which imports `methodtools`. The wrapper loads `alg/LOND.py` via a `methodtools` shim and package stub so the official Python LOND class can run without extra pip installs. Official LOND still:

- treats inputs as p-values (`p = min(1, 1/e)` when wrapping e-values);
- uses finite-horizon renormalized `gamma_series='easy'` = 1/((t+1)(t+2)), **not** UAI `γ_t=1/(t(t+1))`.

Do not treat numerical agreement with UAI e-LOND as a correctness check.

## Donation runtime

Paper states O(log t) per step with an augmented tree. This Python uses the same W̄_t / α_t formula in O(t) per step. Same decisions, worse complexity. Fine for m=200 unit tests and P0; not a runtime claim vs Figure 2.

## Closed r-LOND / donation r-LOND calibrator

Main-text donation r-LOND refers to calibrated e-values E_t = f_t(P_t) without spelling out f_t in the extracted PDF body. This implementation uses E=1/P as the e-LOND correspondence calibrator (not a finite-mean e-value under Uniform). Closed r-LOND uses the closed-e-LOND DP on those e-values with γ/ℓ in the numerator. Labelled independent_reimplementation.

## Not implemented here

Offline donation eBH from `eclosure.experiments` (requires optional extras and is not UAI online donation e-LOND). NYC taxi, m=3000 runtime grid.
