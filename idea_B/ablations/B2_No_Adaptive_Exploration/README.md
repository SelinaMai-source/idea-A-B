# Ablation B2 — replace adaptive exploration with uniform

一次只改分配规则：控制器仍然更新参数并花同一预算，但 `mix_uniform=1`，不再按弱特征值加权。

```bash
cd ~/Research/idea-A-B
PYTHONPATH=. .venv/bin/python idea_B/ablations/B2_No_Adaptive_Exploration/run.py
```
