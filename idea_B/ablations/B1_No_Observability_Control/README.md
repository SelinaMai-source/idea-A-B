# Ablation B1 — disable observability control

一次只改一个组件：关掉基于 \(W_T\) 的观测控制，不再把预算打到弱方向。对照臂是同一环境 B2、同一预算和种子上的完整控制器。

```bash
cd ~/Research/idea-A-B
PYTHONPATH=. .venv/bin/python idea_B/ablations/B1_No_Observability_Control/run.py
```
