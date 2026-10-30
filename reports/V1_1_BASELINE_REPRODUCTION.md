# v1.1 Baseline Reproduction

v1.1 **没有**重跑官方基线。本文件复述 v1 审计 JSON，并说明它们与 v1.1 合成主表不可比。

合成判定（由 TIER2 JSON + 冻结协议）：Idea A = **NOT PROMISING**，Idea B = **NOT PROMISING**。官方复现成败不改变这两条判定。

本地状态对照：AMPLe=`partial`（`0cad341`，`n_rounds=3`），HyPER=`blocked`，Drago=`blocked`（无代码），PP-bandit=`complete`（`d5daeff`，1-D demo），ncvxpp=`partial`（`ccbeedd`，无 MPI）。

五类对象：Published result / Faithful local reproduction / Modified reproduction / Common-protocol comparison / Blocked。

## Idea A 官方基线

machine: `{"os": "macOS 14.5", "arch": "arm64", "model": "MacBook Air M3 16GB", "python_main": "/Users/grantwright/Research/idea-A-B/.venv/bin/python", "python_isolated": "/Users/grantwright/Research/idea-A-B/.isolated_baselines_venv/bin/python", "torch_in_main_venv": false, "disk_free_before_clones": "12Gi", "disk_free_after_clones": "12Gi", "disk_free_after_experiments": "9.7Gi", "isolated_venv_size": "1.3G"}`

### Comparison-based Active Preference Learning for Multi-dimensional Personalization (AMPLe)

- Venue: ACL 2025 Main (Long)
- Official source: https://github.com/ml-postech/AMPLe
- Official code present: `True`
- Upstream commit: `0cad341f6a9e77063ef451c27732bf5c1534c797`
- **local_status: `partial`**
- local_results_path: `idea_A/baselines/AMPLe/official_reproduction/`
- blocker: Full README run is n_rounds=100 with burnin=50000; only n_rounds was shortened to 3. README --dims 0 1 would request minhyeonoh/feedback_assistant_harmlesshelpful_271 (HTTP 401 / not public), so --dims was omitted to load the public 3-aspect dataset. Torch/datasets/wandb/accelerate were installed in an isolated venv, not the main project venv. WANDB_MODE=offline. Acquisition remained volume; burnin remained 50000.
- comparability: NOT DIRECTLY COMPARABLE to Idea A Neyman-audit IPW. AMPLe optimizes multi-dimensional preference identification via volume/random acquisition on RM-scored pairs. Different target, budget, and information.
- what_ran（全部字段来自 JSON）：
```json
{
  "command": "python main.py --seed=1 --task=assistant --rm_names harmless helpful humor --n_rounds=3 --true_preference 0.2 0.7 0.1 --true_beta=10 --beta=2.0 --jitter=0.01 --alpha=10 --init=previous --acquisition=volume --burnin=50000 --gamma=0.0 --mode=likelihood --margin=none --inputs=0",
  "dataset": "minhyeonoh/feedback_assistant_harmlesshelpfulhumor_271",
  "pool_std_diff": 11.289824485778809,
  "pool_noise_rate": 0.04100000113248825,
  "true_preference": [
    0.2,
    0.7,
    0.1
  ],
  "estimation_t0": [
    0.3249063193798065,
    0.32413744926452637,
    0.3509562909603119
  ],
  "estimation_t3": [
    0.0035132216289639473,
    0.9920580983161926,
    0.004428689833730459
  ],
  "l2_to_true_t0": 0.4688851731916284,
  "l2_to_true_t3": 0.3647449275776112,
  "chosen_indices": [
    737,
    95,
    135
  ],
  "wall_sec_approx": 97.8
}
```

### Hybrid Preferences: Learning to Route Instances for Human vs. AI Feedback (HyPER)

- Venue: ACL 2025 Main (Long)
- Official source: https://github.com/allenai/hybrid-preferences
- Official code present: `True`
- Upstream commit: `4d96b27b989828b0b69a430f68d25b6b36e9015a`
- **local_status: `blocked`**
- local_results_path: `idea_A/baselines/HyPER/REPRODUCTION_STATUS.md`
- blocker: Pre-declared. Official pipeline needs Google Cloud TPU v3-128, EasyLM RM/DPO training of Tulu-2 13B / Llama-3.1-8B, vLLM (>=0.4.1; forbidden), Beaker, GCS, and submodules open-instruct + reward-bench (not initialized). Clone without submodules was 632K and was kept. No simplified HyPER was invented.
- comparability: NOT DIRECTLY COMPARABLE. HyPER routes human vs LM annotations to maximize downstream RM / RewardBench scores.

### Online Compatible Reward Identification from Preference Feedback

- Venue: ICML 2026
- Official source: https://proceedings.mlr.press/v306/drago26a.html
- Official code present: `False`
- Upstream commit: `None`
- **local_status: `blocked`**
- local_results_path: `idea_A/baselines/Online_Compatible_Reward_ICML2026/THEORY_COMPARATOR.md`
- blocker: No official code found. PMLR/ICML poster/author PDF have no GitHub badge. github.com/simone-drago has only a github.io page. github.com/marcomussi/SleepingRL is a different ICML 2025 paper. No author implementation was invented.
- comparability: NOT DIRECTLY COMPARABLE. Theory-only contrast: globally compatible reward identification vs Idea A finite-budget audit estimation.

## Idea B 官方基线

machine: `{"os": "macOS 14.5", "arch": "arm64", "model": "MacBook Air M3 16GB", "python_main": "/Users/grantwright/Research/idea-A-B/.venv/bin/python", "python_isolated": "/Users/grantwright/Research/idea-A-B/.isolated_baselines_venv/bin/python", "torch_in_main_venv": false, "disk_free_before_clones": "12Gi", "disk_free_after_clones": "12Gi", "disk_free_after_experiments": "9.7Gi"}`

### Performative Prediction with Bandit Feedback: Learning through Reparameterization

- Venue: ICML 2024
- Official source: https://github.com/UCSC-REAL/PP-bandit-feedback
- Official code present: `True`
- Upstream commit: `d5daeffba8683dc9d8d0f1bedd3fa2c5f168c0c5`
- **local_status: `complete`**
- local_results_path: `idea_B/baselines/Performative_Bandit_ICML2024/official_reproduction/`
- blocker: None
- comparability: Official numerical experiment is a 1-D reparameterized zeroth-order performative-risk demo (Appendix H / Figures 4-6). Comparable only as a performative-prediction algorithmic primitive, not as Idea B observability-controller numbers. License: MIT (REAL@UCSC).
- what_ran（全部字段来自 JSON）：
```json
{
  "command": "/Users/grantwright/Research/idea-A-B/.venv/bin/python official_reproduction/run_official_numerical.py",
  "source": "bandit-icml24.ipynb cells converted; algorithm unchanged",
  "numpy_seed": 0,
  "elapsed_sec": 0.814453125,
  "single_run_final_PR_phi": -0.28164988861568363,
  "single_run_final_PR_theta": -0.2962457875065718,
  "multi_run_PR_phi_final_mean": -0.2883591540722531,
  "multi_run_PR_phi_final_std": 0.0071395956688855965,
  "multi_run_PR_theta_final_mean": -0.291834361799254,
  "multi_run_PR_theta_final_std": 0.007228858237631779
}
```

### Stochastic Optimization Schemes for Performative Prediction with Nonconvex Loss

- Venue: NeurIPS 2024
- Official source: https://github.com/Qiang-CU/nips2024-ncvxpp
- Official code present: `True`
- Upstream commit: `ccbeeddfbf9237aa06dca3a1579529faabd8fe99`
- **local_status: `partial`**
- local_results_path: `idea_B/baselines/Performative_SGD_NeurIPS2024/official_reproduction/`
- blocker: Official full run `cd synthetic-iter && mpiexec -np 10 python main.py` is BLOCKED: mpi4py missing, mpicc/mpiexec not on PATH. A single-process subset of the original synthetic-iter SGD-GD loop was executed with an mpi4py stub (rank=0, size=1). That is still the original per-process algorithm (performative shift + SGD), not a 10-replica paper reproduction. Device was CPU (no CUDA). dataset.py balances Y==1 vs Y==0 but data.csv uses {+1,-1}, so only class +1 survived — official code/data interaction, not a rewrite.
- comparability: NOT a full paper reproduction. Synthetic-iter SGD-GD is a performative linear-model experiment; not the same target as Idea B observability Gramian / closed-loop collapse.
- what_ran（全部字段来自 JSON）：
```json
{
  "command": "PYTHONPATH=official_reproduction/mpi4py_stub python synthetic-iter/main.py",
  "eps_list": [
    0,
    0.1,
    10,
    100
  ],
  "batch": 5,
  "log_maxiter": 5,
  "wall_sec": 540.486,
  "npy": {
    "eps0-batch5-res0.npy": {
      "iter_last": 99426,
      "loss_last": 0.3281349241733551,
      "train_acc_last": 0.9004424778761062,
      "test_acc_last": 1.0,
      "grd_last": 1.559475437673541e-09
    },
    "eps0.1-batch5-res0.npy": {
      "iter_last": 99426,
      "loss_last": 0.3458000421524048,
      "train_acc_last": 0.9004424778761062,
      "test_acc_last": 1.0,
      "grd_last": 8.505186777085783e-09
    },
    "eps10-batch5-res0.npy": {
      "iter_last": 99426,
      "loss_last": 0.3553709387779236,
      "train_acc_last": 0.9004424778761062,
      "test_acc_last": 1.0,
      "grd_last": 2.0597110506059835e-06
    },
    "eps100-batch5-res0.npy": {
      "iter_last": 99426,
      "loss_last": 0.9450080394744873,
      "train_acc_last": 0.09955752212389381,
      "test_acc_last": 0.0,
      "grd_last": 1.3713279123361843e-10
    }
  }
}
```

## 共同协议（不是官方复现）

规则：Do not mix incomparable methods into one leaderboard. Status is Direct / partial / NOT DIRECTLY COMPARABLE. Numbers below are copied from run JSON; none are invented.
n_seeds 说明：n_seeds=3 is preliminary. Paired intervals are not confirmatory.

### `AMPLe_official`

- protocol_label: `partial`
- local_status: `partial`
- comparability status: `NOT DIRECTLY COMPARABLE`
- leaderboard: `official_reproductions_not_on_idea_leaderboards`
- why: AMPLe is query-efficient multi-dimensional preference learning: volume/random acquisition over RM-scored pairs to identify a simplex preference vector. Idea A is identification of finite-population functionals under endogenous censoring with an independent audit channel and known q. Different target, budget, and information. Official AMPLe numbers must not enter the Idea A Neyman/Uniform leaderboard.

### `volume_adapted_ipw`

- protocol_label: `adapted_to_common_protocol`
- local_status: `complete_on_requested_slice`
- comparability status: `partial`
- leaderboard: `idea_A_synthetic_selective_feedback_b0.05_mnar_zero_overlap`
- why: Same synthetic world, budget, estimands, and known-q IPW as Uniform and Neyman, so it may sit on the Idea A common-protocol board as an allocation heuristic. It is only partial as a paper comparator: AMPLe's official volume scores RM-labeled preference pairs under an MH posterior on the simplex and was not reused (would need the HF dataset and torch).

### `HyPER`

- protocol_label: `blocked`
- local_status: `blocked`
- comparability status: `NOT DIRECTLY COMPARABLE`
- leaderboard: `None`
- why: HyPER routes human vs AI annotation to maximize downstream reward-model / RewardBench performance. Idea A audits labels to identify functionals under endogenous censoring. Blocked on this machine (TPU, vLLM, 8B–13B RM training). No simplified HyPER was invented.
- blocker: Pre-declared. Official pipeline needs Google Cloud TPU v3-128, EasyLM RM/DPO training of Tulu-2 13B / Llama-3.1-8B, vLLM (>=0.4.1; forbidden), Beaker, GCS, and submodules open-instruct + reward-bench (not initialized). Clone without submodules was 632K and was kept. No simplified HyPER was invented.

### `Drago_ICML2026`

- protocol_label: `blocked`
- local_status: `blocked`
- comparability status: `NOT DIRECTLY COMPARABLE`
- leaderboard: `None`
- why: Theory comparator only: globally compatible reward identification from preference queries vs Idea A finite-budget audit estimation. No official code. Not an empirical competitor.
- blocker: No official code found. PMLR/ICML poster/author PDF have no GitHub badge. github.com/simone-drago has only a github.io page. github.com/marcomussi/SleepingRL is a different ICML 2025 paper. No author implementation was invented.

### `uniform_ipw`

- protocol_label: `generic_control`
- local_status: `complete_on_requested_slice`
- comparability status: `Direct`
- leaderboard: `idea_A_synthetic_selective_feedback_b0.05_mnar_zero_overlap`
- why: Same world, budget, estimands, and audit IPW as Neyman and the adapted volume heuristic. Uniform q_t = b does not use the natural channel, so for a fixed seed the audit-IPW numbers match across mechanisms (Y is generated before S).

### `neyman_ipw`

- protocol_label: `generic_control`
- local_status: `complete_on_requested_slice`
- comparability status: `Direct`
- leaderboard: `idea_A_synthetic_selective_feedback_b0.05_mnar_zero_overlap`
- why: This is Idea A v1.0. Directly comparable to Uniform and the adapted volume heuristic on the synthetic selective-feedback protocol.

### `IPS_SNIPS_DR_Uncertainty_observed_only_naive_IPS`

- protocol_label: `generic_control`
- local_status: `complete_in_smoke`
- comparability status: `Direct`
- leaderboard: `idea_A_synthetic_smoke_full_matrix (separate from AMPLe/HyPER)`
- why: Same synthetic protocol as Neyman/Uniform. Already run in idea_A/results (84 runs). Not re-executed here. Not published paper reproductions.

### `PP-bandit_official`

- protocol_label: `official_reproduction`
- local_status: `complete`
- comparability status: `NOT DIRECTLY COMPARABLE`
- leaderboard: `official_reproductions_not_on_idea_leaderboards`
- why: Official numerical experiment is a 1-D reparameterized zeroth-order performative-risk demo (Appendix H). It has decision-dependent distributions but no Idea B skill-wise feedback censoring, no observability Gramian, and no audit budget. Keep as an algorithmic primitive table, not on the B2 observability leaderboard.

### `pp_bandit_zeroth_order_reparam_adapted`

- protocol_label: `adapted-skipped`
- local_status: `BLOCKED`
- comparability status: `NOT DIRECTLY COMPARABLE`
- leaderboard: `None`
- why: Official PP-bandit core (bandit-icml24.ipynb / official_reproduction/run_official_numerical.py) is a 1-D reparameterization x = y^2 with two-point zeroth-order updates and an inner LearnModel that minimizes Bernoulli KL(x_target || y^2). That procedure queries an uncensored scalar performative-risk function. Idea B environment B2 is a 3-D linear skill model whose performative channel is skill-wise observation censoring, not a Bernoulli mean map, and labels are revealed only under a limited audit budget. A faithful LearnModel+reparameterization port is not feasible on this NumPy protocol without rewriting the paper's 1-D Bernoulli experiment. A two-point ZO-SGD with uniform audits would be a different generic optimizer and must not be relabeled as adapted PP-bandit. Official numerical demo remains in idea_B/baselines/Performative_Bandit_ICML2024/official_reproduction/ (COMPLETE, separate from this leaderboard).

### `ncvxpp_Performative_SGD`

- protocol_label: `partial`
- local_status: `partial`
- comparability status: `NOT DIRECTLY COMPARABLE`
- leaderboard: `official_reproductions_not_on_idea_leaderboards`
- why: Performative SGD / SGD-GD is decision-dependent risk for a linear classifier. It does not implement Idea B's skill-wise feedback censoring or observability Gramian. Official full mpiexec -np 10 run is blocked (no mpi4py). Single-replica synthetic-iter is partial, not a 10-replica paper reproduction. Do not relabel Idea B Greedy/Lazy as official SGD-GD.
- blocker: Official full run `cd synthetic-iter && mpiexec -np 10 python main.py` is BLOCKED: mpi4py missing, mpicc/mpiexec not on PATH. A single-process subset of the original synthetic-iter SGD-GD loop was executed with an mpi4py stub (rank=0, size=1). That is still the original per-process algorithm (performative shift + SGD), not a 10-replica paper reproduction. Device was CPU (no CUDA). dataset.py balances Y==1 vs Y==0 but data.csv uses {+1,-1}, so only class +1 survived — official code/data interaction, not a rewrite.

### `ObservabilityController`

- protocol_label: `idea_B_v1.0`
- local_status: `complete_on_requested_slice`
- comparability status: `Direct`
- leaderboard: `idea_B_B2_budget0.05_common_protocol`
- why: Native method on environment B2 with the same 5% budget and seeds as Uniform and Greedy.

### `UniformExploration`

- protocol_label: `generic_control`
- local_status: `complete_on_requested_slice`
- comparability status: `Direct`
- leaderboard: `idea_B_B2_budget0.05_common_protocol`
- why: Same B2 closed loop, budget, and metrics as the observability controller.

### `GreedyDeployment`

- protocol_label: `generic_control`
- local_status: `complete_on_requested_slice`
- comparability status: `Direct`
- leaderboard: `idea_B_B2_budget0.05_common_protocol`
- why: Same B2 protocol as Uniform and the observability controller. NOT an official Performative SGD / SGD-GD reproduction. LazyDeployment is the same generic family and was not re-run in this slice.

### `LazyDeployment_StandardOnlineSGD`

- protocol_label: `generic_control`
- local_status: `complete_in_smoke`
- comparability status: `Direct`
- leaderboard: `idea_B_synthetic_smoke (separate from official PP/SGD tables)`
- why: Already present in idea_B/results as generic controls. Direct on the Idea B protocol. Must not be relabeled as official SGD-GD or PP-bandit. Not part of the 9-run common-protocol slice.

不得把 AMPLe / HyPER / PP-bandit / ncvxpp 的数字排进 v1.1 Uniform DR 或 E-optimal 主表。
