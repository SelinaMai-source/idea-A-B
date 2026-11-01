# Idea C

**Current Idea C v0 Experimental Results:** [`v0实验版本+结果.md`](v0实验版本+结果.md)

v0 只忠实复现四篇 2026 论文，并跑两个 Common Benchmark。不开发 C-v1 新方法。数字只来自 `results/v0/raw/` 的 JSON/CSV；未跑格子是 `NOT_RUN`，不是 0。

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
