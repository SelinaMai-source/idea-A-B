# Ablation B4 — remove policy-dependent feedback

把 B2 的闭环观测通道换成 B1 的外生缺失，控制器和预算不变。用来检验“策略诱发”是否真的来自闭环，而不是来自任何缺失。

```bash
cd ~/Research/idea-A-B
PYTHONPATH=. .venv/bin/python idea_B/ablations/B4_No_Policy_Feedback_Loop/run.py
```
