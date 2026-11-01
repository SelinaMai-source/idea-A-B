# eclosure official clone manifest

- **Intended paper:** Improving online FDR procedures via online analogs of e-closure and compound e-values (Xu, Fischer, Ramdas)
- **Intended venue:** UAI 2026 (**NOT CCF-A Main**)
- **PMLR:** https://proceedings.mlr.press/v337/xu26a.html
- **PDF:** https://raw.githubusercontent.com/mlresearch/v337/main/assets/xu26a/xu26a.pdf
- **Cloned repo:** https://github.com/neilzxu/eclosure
- **What this repo actually is:** official code for the *offline* e-Closure paper *Bringing Closure to False Discovery Rate Control* (Xu et al., arXiv:2509.02517). Public API is `closedeBH` / `closedBY` (Python/R wrapping C++). Homepage on GitHub points to that arXiv, **not** UAI 2026.
- **UAI 2026 online closed / donation e-LOND:** no dedicated public repository found at audit time. The UAI PDF says accompanying code exists, but the only public `neilzxu/eclosure` tree is the offline package. Treat UAI online methods as `independent_reimplementation` unless a matching repo appears later.
- **Clone method:** `git clone --depth 1 --recurse-submodules=no https://github.com/neilzxu/eclosure.git idea_C/baselines/eclosure/official`
- **Upstream commit:** `8bbd5a37b48abb6d79f5e1eef95d4fe6315339b5`
- **Commit date:** 2026-03-23 17:05:48 -0400
- **Commit subject:** version update
- **Branch / tag:** `release` / `v0.1.1b`
- **Shallow:** yes
- **On-disk size after shallow clone:** 3.6M
- **Clone date (local):** 2026-11-01
- **Extras initialized:** no (`pip install .[experiments]` / plotting / docs **not** run)
- **Parent-repo policy:** this manifest is the committed record. The clone keeps its own `.git` and is gitignored. Do not vendor the tree.
- **Local path:** `idea_C/baselines/eclosure/official/`
