# Idea C

**Current Idea C v0 Experimental Results:** [`v0实验版本+结果.md`](v0实验版本+结果.md)

v0 只忠实复现四篇 2026 论文，并跑两个 Common Benchmark。不开发 C-v1 新方法。数字只来自 `results/v0/raw/` 的 JSON/CSV；未跑格子是 `NOT_RUN`，不是 0。

## v0.1 全文复现入口

官方仓确认不存在。模式只有 independent。UAI **不是** CCF-A Main。e-PS **是预印本**。`--scale paper` 必须另加 `--paper-scale`。30-seed 不是全文复现。v0.1 原始结果写 `results/v0.1/raw/`，不覆盖 v0 raw CSV。

```bash
cd /Users/grantwright/Desktop/科研/10.30新课题/v0
export PYTHONPATH="$PWD"
.venv/bin/python idea_C/scripts/run_v0_1.py --help
.venv/bin/python idea_C/scripts/run_v0_1.py --list
.venv/bin/python idea_C/scripts/run_v0_1.py --paper score --figure 1 --scale smoke
.venv/bin/python idea_C/scripts/run_v0_1.py --paper aaai --benchmark aaai_robust_gaussian --scale smoke --run
.venv/bin/python idea_C/scripts/run_v0_1.py --mode official   # 应报错：无官方仓
.venv/bin/python idea_C/scripts/run_v0_1.py --scale paper     # 应报错：缺 --paper-scale
.venv/bin/python idea_C/scripts/verify_paper_results.py
```

清单与容差：[`reports/v0.1/FULL_REPRODUCTION_CHECKLIST.md`](reports/v0.1/FULL_REPRODUCTION_CHECKLIST.md)。官方代码审计：[`reports/v0.1/OFFICIAL_CODE_AUDIT.md`](reports/v0.1/OFFICIAL_CODE_AUDIT.md)。对照表：[`reports/v0.1/PAPER_VS_REPRODUCTION.md`](reports/v0.1/PAPER_VS_REPRODUCTION.md)。状态：[`reports/v0.1/REPRODUCTION_STATUS.md`](reports/v0.1/REPRODUCTION_STATUS.md)。对照图：[`results/v0.1/figures/`](results/v0.1/figures/)。

P1 overlap（只在 P0 可信后重跑，标 `ADAPTED_PROTOCOL` / `C-v0.1_RERUN`；永不 500）：

```bash
.venv/bin/python idea_C/scripts/run_p1a.py --v01 --setting clean --scale smoke --run
.venv/bin/python idea_C/scripts/run_p1a.py --v01 --setting contaminated --scale smoke --run
.venv/bin/python idea_C/scripts/run_p1b.py --v01 --scale smoke --run --assume-cpqd
.venv/bin/python idea_C/scripts/run_p1b.py --v01 --scale pilot --run --assume-cpqd --skip-closed
```

| 论文 | Venue | v0 策略 |
| --- | --- | --- |
| On the Robustness of Bandit Multiple Testing (Zhou & Liu) | AAAI 2026 Main | `independent_reimplementation` |
| SCORE (Kuang, Gang, Xia) | ICML 2026 Main | `independent_reimplementation` |
| e-Closure / donation (Xu, Fischer, Ramdas) | UAI 2026（**非 CCF-A Main**） | 官方离线仓已浅克隆；UAI 在线方法独立实现 |
| e-PS (Lin, Ma, Ren, Wei) | arXiv:2609.26651 **预印本**（非已发表 CCF-A） | `independent_reimplementation`；不做 LLM watermark |

## 环境

MacBook Air M3，16GB，CPU。使用仓库根 `.venv`（NumPy/SciPy）。不装 vLLM/CUDA。

```bash
cd /Users/grantwright/Desktop/科研/10.30新课题/v0
export PYTHONPATH="$PWD"
.venv/bin/python idea_C/scripts/run_v0.py --list
.venv/bin/python idea_C/scripts/update_v0_results_md.py
.venv/bin/python -m pytest idea_C/tests -q
```

`run_v0.py` 在算法包仍为空时拒绝 Monte Carlo。先 unit test → 3-seed smoke → 30-seed pilot；原文 200/500 不得用 30-seed 冒充 paper-scale。

## 产物

- 总表：`v0实验版本+结果.md`（每 benchmark 独立表，只追加）
- raw：`results/v0/raw/<run_id>/`
- 审计：`reports/v0/BASELINE_AUDIT.md`
- 笔记：`研究记录.md`，`论文阅读笔记.md`
