# MultiPref subset manifest

- **Source:** `allenai/multipref`
- **Pinned revision:** `12910233a0238a997ebe425656e9dfed7b0ff031`
- **Download source:** `huggingface`
- **Dataset version stated by upstream:** MultiPref 1.0
- **This file is the Stage 6 feasibility subset, not a training set.**

## What was kept

The `default` config keeps fine-grained annotations:

- `normal_worker_annotations`: two crowdworker annotations (human)
- `expert_worker_annotations`: two domain-expert annotations (human)
- `model_generated_gpt4`: joined from `gpt4_overall_binarized` by `comparison_id`

Preference dimensions are preserved separately: helpful, truthful, harmless, overall.
Disagreements are recorded in each row's `disagreements` field.

## What was not done

- `human_overall_binarized` was **not** used. That config majority-votes humans and
  randomly breaks ties; it is a collapsed view, not unbiased ground truth.
- No mixture of human / expert / GPT-4 labels into a single target.
- No reward-model training and no Idea A/B synthetic-matrix rerun.

## Subset rule

Sort all default-config rows by comparison_id (lexicographic). Walk in that order and keep the first 50 unique prompt_id values. Include every comparison whose prompt_id is in that set, so a prompt is never split across train/val/test. Split the selected prompt_ids in first-seen order as {'train': 35, 'val': 8, 'test': 7}.

| split | prompts | comparisons |
| --- | ---: | ---: |
| train | 35 | 70 |
| val | 8 | 15 |
| test | 7 | 14 |
| **total** | 50 | 99 |

- Committable subset JSON: `subset.json` (1501152 bytes, sha256 `8eba3a14d0d6a03c4257c09ad618345bc758682642a1b5e74e5760e010f1ba9c`)
- IDs / splits: `subset_ids.json`
- Schema check: `SCHEMA_VALIDATION.json`
- Rebuild: `python scripts/download_benchmarks.py`

Raw parquet stays in gitignored `/Users/grantwright/Research/idea-A-B/.cache/datasets/allenai_multipref/12910233a0238a997ebe425656e9dfed7b0ff031`.
