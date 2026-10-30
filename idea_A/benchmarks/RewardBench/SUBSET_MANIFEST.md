# RewardBench schema check

- **Source:** `allenai/reward-bench`
- **Pinned revision:** `168d848cdbbea9764fae4a544dc9ca1e6cca4931`
- **Version:** RewardBench v1 (Lambert et al., 2024). This is **not** RewardBench-2.
- **Download source:** `huggingface`
- **Official eval split used for schema:** `filtered` (n=2985); `raw` n=5123

## Schema

Required fields observed: `prompt`, `chosen`, `rejected`, `chosen_model`, `rejected_model`, `subset`, `id`.
On this revision, `chosen` and `rejected` are strings (single-turn pairs).

Tiny committable sample: first 50 filtered rows by integer `id`
(`schema_sample.json`, 160320 bytes).

## What was not done

- No comparison to the published RewardBench leaderboard.
- No reward-model scoring. There is no compatible small RM scorer in this repo.
  Qwen2.5-0.5B-Instruct is not an official RewardBench reward model.
- Did not re-clone https://github.com/allenai/reward-bench. HyPER lists it as a
  submodule at `idea_A/baselines/HyPER/reward-bench` and that tree was intentionally
  left uninitialized.

Rebuild: `python scripts/download_benchmarks.py`
