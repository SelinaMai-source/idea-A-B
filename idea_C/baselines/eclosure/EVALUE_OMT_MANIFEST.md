# evalue-omt clone manifest

- **Repo:** https://github.com/neilzxu/evalue-omt
- **What this repo actually is:** official code for *Online multiple testing with e-values* (Xu, Wang, Ramdas), AISTATS 2024, arXiv:2311.06412. **Not** the UAI 2026 e-closure paper.
- **Why cloned:** the UAI 2026 plan lists it as related official source for e-LOND / local-dependence simulations (`lag_comp`). Useful as a baseline reference, not as a drop-in UAI 2026 closed/donation implementation.
- **Clone method:** `git clone --depth 1 --recurse-submodules=no https://github.com/neilzxu/evalue-omt.git idea_C/baselines/eclosure/evalue-omt`
- **Upstream commit:** `cda6e9962514924044143920cf1cadbec8738d52`
- **Commit date:** 2024-03-08 23:55:23 +0000
- **Commit subject:** Initial code release
- **Branch:** `publishable`
- **Shallow:** yes
- **On-disk size after shallow clone:** 4.1M
- **Clone date (local):** 2026-11-01
- **Extras initialized:** no (DeepPurpose / conda env **not** installed)
- **Paper experiment names in README:** `lag_comp`, `wor_comp`
- **Parent-repo policy:** this manifest is the committed record. The clone keeps its own `.git` and is gitignored.
- **Local path:** `idea_C/baselines/eclosure/evalue-omt/`
- **v0 wrapper:** `official_wrappers.aistats_elond_on_evalues` loads official `alg.LOND` (AISTATS 2024) with a `methodtools` shim. Finite-horizon `easy` γ ≠ UAI `γ_t=1/(t(t+1))`. `lag_comp` is the related official experiment name; P0 local-dependence uses UAI Section 4 DGP. See `DEVIATION.md`.
