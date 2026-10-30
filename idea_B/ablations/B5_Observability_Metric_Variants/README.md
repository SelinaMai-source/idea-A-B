# Ablation B5 — compare information metrics

同一控制器和预算，只换分配用的信息读出：\(\lambda_{\min}\)（默认）、\(\mathrm{tr}(W^{-1})\)、以及 logdet 代理。不把这些矩阵改名叫别的 Gramian。

```bash
cd ~/Research/idea-A-B
PYTHONPATH=. .venv/bin/python idea_B/ablations/B5_Observability_Metric_Variants/run.py
```
