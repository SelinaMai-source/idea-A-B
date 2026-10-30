#!/usr/bin/env python3
"""Download a conservative MultiPref subset and RewardBench schema sample.

This script does not train models, does not score RewardBench leaderboards,
and does not collapse MultiPref annotation types into one ground truth.

Run with the isolated venv that already has `datasets` / `huggingface_hub`:

    .isolated_baselines_venv/bin/python scripts/download_benchmarks.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
HF_HOME = ROOT / ".cache" / "hf"
DATA_CACHE = ROOT / ".cache" / "datasets"
MIN_FREE_BYTES = 4 * 1024**3
# Upper bounds used only for the pre-download disk guard.
MULTIPREF_BUDGET_BYTES = 600 * 1024 * 1024
REWARDBENCH_BUDGET_BYTES = 150 * 1024 * 1024

MULTIPREF_REPO = "allenai/multipref"
MULTIPREF_REVISION = "12910233a0238a997ebe425656e9dfed7b0ff031"
REWARD_BENCH_REPO = "allenai/reward-bench"
REWARD_BENCH_REVISION = "168d848cdbbea9764fae4a544dc9ca1e6cca4931"

TARGET_N_PROMPTS = 50
SPLIT_COUNTS = {"train": 35, "val": 8, "test": 7}
REWARD_BENCH_SAMPLE_N = 50

PREF_KEYS = ("helpful_pref", "truthful_pref", "harmless_pref", "overall_pref")
MULTIPREF_REQUIRED = (
    "comparison_id",
    "prompt_id",
    "text",
    "completion_a",
    "model_a",
    "completion_b",
    "model_b",
    "source",
    "category",
    "subject_study",
    "highest_level_degree",
    "normal_worker_annotations",
    "expert_worker_annotations",
)
REWARD_BENCH_REQUIRED = (
    "prompt",
    "chosen",
    "rejected",
    "chosen_model",
    "rejected_model",
    "subset",
    "id",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def disk_snapshot(path: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    return {
        "path": str(path),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "free_gib": usage.free / (1024**3),
        "recorded_at_utc": utc_now(),
    }


def require_disk(path: Path, extra_needed: int, label: str) -> dict[str, Any]:
    snap = disk_snapshot(path)
    remaining_after = snap["free_bytes"] - extra_needed
    snap["label"] = label
    snap["extra_needed_bytes"] = extra_needed
    snap["projected_free_bytes"] = remaining_after
    if remaining_after < MIN_FREE_BYTES:
        raise RuntimeError(
            f"DISK_GUARD skip {label}: free={snap['free_bytes']} extra={extra_needed} "
            f"projected_free={remaining_after} < {MIN_FREE_BYTES}"
        )
    return snap


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if hasattr(obj, "isoformat"):
        try:
            return obj.isoformat()
        except Exception:
            return str(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", "replace")
    module = getattr(type(obj), "__module__", "")
    if module.startswith("numpy"):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    return obj


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def snapshot_repo(
    repo_id: str,
    revision: str,
    local_dir: Path,
    repo_type: str,
    allow_patterns: list[str],
) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    before = disk_snapshot(ROOT)
    try:
        resolved = snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            local_dir=str(local_dir),
            allow_patterns=allow_patterns,
        )
        source = "huggingface"
        error = None
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        source, resolved = _try_modelscope_mirror(
            repo_id=repo_id,
            revision=revision,
            local_dir=local_dir,
            repo_type=repo_type,
            allow_patterns=allow_patterns,
        )
        if source is None:
            raise RuntimeError(
                f"HF download failed for {repo_id}@{revision}: {error}; "
                f"ModelScope fallback also failed or was not used."
            ) from exc
    after = disk_snapshot(ROOT)
    return {
        "repo_id": repo_id,
        "revision": revision,
        "repo_type": repo_type,
        "local_dir": str(local_dir),
        "resolved_path": str(resolved),
        "source": source,
        "hf_error": error,
        "disk_before": before,
        "disk_after": after,
        "allow_patterns": allow_patterns,
    }


def _try_modelscope_mirror(
    repo_id: str,
    revision: str,
    local_dir: Path,
    repo_type: str,
    allow_patterns: list[str],
) -> tuple[str | None, Path | None]:
    """Attempt ModelScope only as a verified mirror after Hugging Face failure."""
    try:
        import modelscope  # type: ignore
    except Exception:
        print(
            "ModelScope SDK is not installed. Not installing it preemptively. "
            "HF failed; cannot use ModelScope without an extra package install.",
            file=sys.stderr,
        )
        return None, None
    try:
        from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot

        ms_id = repo_id
        resolved = ms_snapshot(
            ms_id,
            revision=revision,
            cache_dir=str(local_dir),
            allow_patterns=allow_patterns,
        )
        return "modelscope", Path(resolved)
    except Exception as exc:
        print(f"ModelScope fallback failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return None, None


def load_parquet_rows(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    return [jsonable(row) for row in table.to_pylist()]


def find_parquet(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if not matches:
        matches = sorted(root.rglob(Path(pattern).name))
    if not matches:
        raise FileNotFoundError(f"No parquet matching {pattern} under {root}")
    return matches[0]


def validate_multipref_row(row: dict[str, Any]) -> list[str]:
    errors = []
    for key in MULTIPREF_REQUIRED:
        if key not in row:
            errors.append(f"missing field {key}")
    for field in ("normal_worker_annotations", "expert_worker_annotations"):
        anns = row.get(field)
        if not isinstance(anns, list) or len(anns) != 2:
            errors.append(f"{field} is not a length-2 list")
            continue
        for i, ann in enumerate(anns):
            if not isinstance(ann, dict):
                errors.append(f"{field}[{i}] is not a dict")
                continue
            for pref_key in PREF_KEYS:
                if pref_key not in ann:
                    errors.append(f"{field}[{i}] missing {pref_key}")
            if "evaluator" not in ann:
                errors.append(f"{field}[{i}] missing evaluator")
    return errors


def disagreement_record(row: dict[str, Any]) -> dict[str, Any]:
    normals = row.get("normal_worker_annotations") or []
    experts = row.get("expert_worker_annotations") or []
    gpt4 = row.get("model_generated_gpt4")
    by_dim: dict[str, Any] = {}
    for key in PREF_KEYS:
        n_vals = [a.get(key) for a in normals if isinstance(a, dict)]
        e_vals = [a.get(key) for a in experts if isinstance(a, dict)]
        rec = {
            "normal_values": n_vals,
            "expert_values": e_vals,
            "normal_internal_agree": len(set(n_vals)) == 1 if n_vals else None,
            "expert_internal_agree": len(set(e_vals)) == 1 if e_vals else None,
            "normal_vs_expert_value_sets_equal": set(n_vals) == set(e_vals) if n_vals and e_vals else None,
        }
        if isinstance(gpt4, dict) and key == "overall_pref":
            rec["gpt4_binarized"] = {
                "chosen_model": gpt4.get("chosen_model"),
                "rejected_model": gpt4.get("rejected_model"),
                "tie_is_common": gpt4.get("tie_is_common"),
            }
        by_dim[key] = rec
    gpt4_view = None
    if isinstance(gpt4, dict):
        gpt4_view = {
            "config": "gpt4_overall_binarized",
            "annotator_type": "model_generated",
            "model": gpt4.get("_model") or "gpt-4-turbo-2024-04-09",
            "chosen_model": gpt4.get("chosen_model"),
            "rejected_model": gpt4.get("rejected_model"),
            "tie_is_common": gpt4.get("tie_is_common"),
            "has_likert_overall_pref": "overall_pref" in gpt4,
            "note": (
                "This config is a binarized chosen/rejected view, not a Likert "
                "overall_pref. It is stored separately from human annotations."
            ),
        }
    return {
        "normal_workers_disagree_overall": by_dim["overall_pref"]["normal_internal_agree"] is False,
        "experts_disagree_overall": by_dim["overall_pref"]["expert_internal_agree"] is False,
        "human_group_value_sets_differ_overall": by_dim["overall_pref"]["normal_vs_expert_value_sets_equal"] is False,
        "gpt4_binarized_view": gpt4_view,
        "by_dimension": by_dim,
        "note": "Disagreements are preserved. No majority vote or mixed-source ground truth is created.",
    }


def attach_gpt4(default_rows: list[dict[str, Any]], gpt4_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_cid: dict[str, dict[str, Any]] = {}
    for row in gpt4_rows:
        cid = str(row.get("comparison_id") or row.get("id") or "")
        if cid:
            by_cid[cid] = row
    matched = 0
    for row in default_rows:
        cid = str(row.get("comparison_id") or "")
        gpt4 = by_cid.get(cid)
        row["annotation_provenance"] = {
            "human_normal_crowdworkers": "normal_worker_annotations",
            "human_domain_experts": "expert_worker_annotations",
            "model_generated_gpt4": "model_generated_gpt4",
            "collapsed_human_majority_not_used": True,
        }
        if gpt4 is None:
            row["model_generated_gpt4"] = None
            continue
        payload = dict(gpt4)
        payload["_source_config"] = "gpt4_overall_binarized"
        payload["_annotator_type"] = "model_generated"
        payload["_model"] = "gpt-4-turbo-2024-04-09"
        row["model_generated_gpt4"] = payload
        matched += 1
    return {
        "gpt4_rows": len(gpt4_rows),
        "default_rows": len(default_rows),
        "matched_by_comparison_id": matched,
        "gpt4_columns": sorted({k for row in gpt4_rows for k in row.keys()}),
    }


def select_multipref_subset(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ordered = sorted(rows, key=lambda r: str(r.get("comparison_id") or ""))
    selected_prompt_ids: list[str] = []
    selected_set: set[str] = set()
    for row in ordered:
        pid = str(row.get("prompt_id") or "")
        if not pid or pid in selected_set:
            continue
        selected_set.add(pid)
        selected_prompt_ids.append(pid)
        if len(selected_prompt_ids) >= TARGET_N_PROMPTS:
            break
    prompt_to_split: dict[str, str] = {}
    n_train, n_val, n_test = SPLIT_COUNTS["train"], SPLIT_COUNTS["val"], SPLIT_COUNTS["test"]
    for i, pid in enumerate(selected_prompt_ids):
        if i < n_train:
            prompt_to_split[pid] = "train"
        elif i < n_train + n_val:
            prompt_to_split[pid] = "val"
        else:
            prompt_to_split[pid] = "test"
    subset = []
    for row in ordered:
        pid = str(row.get("prompt_id") or "")
        if pid not in selected_set:
            continue
        item = dict(row)
        item["split"] = prompt_to_split[pid]
        item["disagreements"] = disagreement_record(item)
        subset.append(item)
    counts = {"train": 0, "val": 0, "test": 0}
    prompt_counts = {"train": set(), "val": set(), "test": set()}
    for item in subset:
        counts[item["split"]] += 1
        prompt_counts[item["split"]].add(item["prompt_id"])
    meta = {
        "selection_rule": (
            "Sort all default-config rows by comparison_id (lexicographic). "
            f"Walk in that order and keep the first {TARGET_N_PROMPTS} unique prompt_id values. "
            "Include every comparison whose prompt_id is in that set, so a prompt is never split "
            "across train/val/test. Split the selected prompt_ids in first-seen order as "
            f"{SPLIT_COUNTS}."
        ),
        "n_source_rows": len(rows),
        "n_selected_prompt_ids": len(selected_prompt_ids),
        "n_selected_comparisons": len(subset),
        "prompt_ids_in_first_seen_order": selected_prompt_ids,
        "examples_by_split": counts,
        "prompts_by_split": {k: len(v) for k, v in prompt_counts.items()},
        "no_collapsed_ground_truth": True,
    }
    return subset, meta


def build_multipref(out_dir: Path) -> dict[str, Any]:
    require_disk(ROOT, MULTIPREF_BUDGET_BYTES, "multipref")
    cache_dir = DATA_CACHE / "allenai_multipref" / MULTIPREF_REVISION
    download_info = snapshot_repo(
        repo_id=MULTIPREF_REPO,
        revision=MULTIPREF_REVISION,
        local_dir=cache_dir,
        repo_type="dataset",
        allow_patterns=[
            "README.md",
            "data/*.parquet",
            "gpt4_overall_binarized/*.parquet",
        ],
    )
    default_parquet = find_parquet(cache_dir, "data/*.parquet")
    gpt4_parquet = find_parquet(cache_dir, "gpt4_overall_binarized/*.parquet")
    default_rows = load_parquet_rows(default_parquet)
    gpt4_rows = load_parquet_rows(gpt4_parquet)
    gpt4_join = attach_gpt4(default_rows, gpt4_rows)
    schema_errors = []
    for i, row in enumerate(default_rows[:20]):
        schema_errors.extend(f"row{i}: {e}" for e in validate_multipref_row(row))
    subset, subset_meta = select_multipref_subset(default_rows)
    subset_errors = []
    for i, row in enumerate(subset):
        subset_errors.extend(f"subset{i}: {e}" for e in validate_multipref_row(row))
        if row.get("annotation_provenance", {}).get("collapsed_human_majority_not_used") is not True:
            subset_errors.append(f"subset{i}: provenance collapsed unexpectedly")

    ids_payload = {
        "revision": MULTIPREF_REVISION,
        "repo_id": MULTIPREF_REPO,
        "selection": subset_meta,
        "comparison_ids": [row["comparison_id"] for row in subset],
        "prompt_ids": sorted({row["prompt_id"] for row in subset}),
        "splits": {
            split: [row["comparison_id"] for row in subset if row["split"] == split]
            for split in ("train", "val", "test")
        },
    }
    write_json(out_dir / "subset.json", subset)
    write_json(out_dir / "subset_ids.json", ids_payload)
    digest = hashlib.sha256((out_dir / "subset.json").read_bytes()).hexdigest()
    schema = {
        "ok": not schema_errors and not subset_errors,
        "required_fields": list(MULTIPREF_REQUIRED),
        "preference_dimensions": list(PREF_KEYS),
        "annotation_types_preserved": [
            "human_normal_crowdworkers",
            "human_domain_experts",
            "model_generated_gpt4",
        ],
        "collapsed_configs_not_used": ["human_overall_binarized"],
        "n_source_default_rows": len(default_rows),
        "n_source_gpt4_rows": len(gpt4_rows),
        "head20_schema_errors": schema_errors,
        "subset_schema_errors": subset_errors,
        "gpt4_join": gpt4_join,
        "default_parquet": str(default_parquet),
        "gpt4_parquet": str(gpt4_parquet),
        "subset_sha256": digest,
        "subset_bytes": (out_dir / "subset.json").stat().st_size,
    }
    write_json(out_dir / "SCHEMA_VALIDATION.json", schema)
    manifest = f"""# MultiPref subset manifest

- **Source:** `{MULTIPREF_REPO}`
- **Pinned revision:** `{MULTIPREF_REVISION}`
- **Download source:** `{download_info["source"]}`
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

{subset_meta["selection_rule"]}

| split | prompts | comparisons |
| --- | ---: | ---: |
| train | {subset_meta["prompts_by_split"]["train"]} | {subset_meta["examples_by_split"]["train"]} |
| val | {subset_meta["prompts_by_split"]["val"]} | {subset_meta["examples_by_split"]["val"]} |
| test | {subset_meta["prompts_by_split"]["test"]} | {subset_meta["examples_by_split"]["test"]} |
| **total** | {subset_meta["n_selected_prompt_ids"]} | {subset_meta["n_selected_comparisons"]} |

- Committable subset JSON: `subset.json` ({schema["subset_bytes"]} bytes, sha256 `{digest}`)
- IDs / splits: `subset_ids.json`
- Schema check: `SCHEMA_VALIDATION.json`
- Rebuild: `python scripts/download_benchmarks.py`

Raw parquet stays in gitignored `{cache_dir}`.
"""
    (out_dir / "SUBSET_MANIFEST.md").write_text(manifest, encoding="utf-8")
    loader = '''#!/usr/bin/env python3
"""Load the committed MultiPref subset. Does not invent a single ground-truth label."""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_subset(split=None):
    rows = json.loads((HERE / "subset.json").read_text(encoding="utf-8"))
    if split is None:
        return rows
    return [row for row in rows if row.get("split") == split]


def load_ids():
    return json.loads((HERE / "subset_ids.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    data = load_subset()
    print(f"n={len(data)} splits={{k: sum(1 for r in data if r['split']==k) for k in ('train','val','test')}}")
    print("annotation types preserved: normal_worker_annotations, expert_worker_annotations, model_generated_gpt4")
'''
    (out_dir / "load_subset.py").write_text(loader, encoding="utf-8")
    return {
        "status": "ok" if schema["ok"] else "schema_errors",
        "download": download_info,
        "subset_meta": subset_meta,
        "schema": {
            "ok": schema["ok"],
            "head20_schema_errors": schema_errors,
            "subset_schema_errors": subset_errors,
            "subset_bytes": schema["subset_bytes"],
            "subset_sha256": digest,
            "gpt4_join": gpt4_join,
        },
        "paths": {
            "subset_json": str(out_dir / "subset.json"),
            "subset_ids": str(out_dir / "subset_ids.json"),
            "manifest": str(out_dir / "SUBSET_MANIFEST.md"),
        },
    }


def validate_rewardbench_row(row: dict[str, Any]) -> list[str]:
    errors = []
    for key in REWARD_BENCH_REQUIRED:
        if key not in row:
            errors.append(f"missing field {key}")
    for key in ("prompt", "chosen", "rejected"):
        if key in row and not isinstance(row[key], str):
            errors.append(f"{key} is {type(row[key]).__name__}, expected str")
    return errors


def build_rewardbench(out_dir: Path) -> dict[str, Any]:
    require_disk(ROOT, REWARDBENCH_BUDGET_BYTES, "rewardbench")
    cache_dir = DATA_CACHE / "allenai_reward_bench" / REWARD_BENCH_REVISION
    download_info = snapshot_repo(
        repo_id=REWARD_BENCH_REPO,
        revision=REWARD_BENCH_REVISION,
        local_dir=cache_dir,
        repo_type="dataset",
        allow_patterns=[
            "README.md",
            "data/filtered-*.parquet",
            "data/raw-*.parquet",
        ],
    )
    filtered_parquet = find_parquet(cache_dir, "data/filtered-*.parquet")
    raw_parquet = find_parquet(cache_dir, "data/raw-*.parquet")
    filtered_rows = load_parquet_rows(filtered_parquet)
    raw_rows = load_parquet_rows(raw_parquet)
    filtered_rows_sorted = sorted(filtered_rows, key=lambda r: int(r.get("id", 10**12)))
    sample = filtered_rows_sorted[:REWARD_BENCH_SAMPLE_N]
    errors = []
    for i, row in enumerate(sample):
        errors.extend(f"sample{i}: {e}" for e in validate_rewardbench_row(row))
    for i, row in enumerate(filtered_rows_sorted[:20]):
        errors.extend(f"filtered{i}: {e}" for e in validate_rewardbench_row(row))
    subsets = {}
    for row in filtered_rows:
        subsets[str(row.get("subset"))] = subsets.get(str(row.get("subset")), 0) + 1
    hyper_placeholder = ROOT / "idea_A" / "baselines" / "HyPER" / "reward-bench"
    hyper_note = {
        "official_code": "https://github.com/allenai/reward-bench",
        "hyper_submodule_path": str(hyper_placeholder),
        "hyper_submodule_initialized": hyper_placeholder.exists() and any(hyper_placeholder.iterdir())
        if hyper_placeholder.exists()
        else False,
        "action": "Did not re-clone the RewardBench git tree. Schema check against allenai/reward-bench is enough.",
    }
    sample_path = out_dir / "schema_sample.json"
    write_json(sample_path, sample)
    digest = hashlib.sha256(sample_path.read_bytes()).hexdigest()
    schema = {
        "ok": not errors,
        "dataset_name": REWARD_BENCH_REPO,
        "rewardbench_version": "v1 (Lambert et al. 2024 RewardBench; NOT RewardBench-2 / allenai/reward-bench-2)",
        "pinned_revision": REWARD_BENCH_REVISION,
        "splits_observed": {
            "filtered": len(filtered_rows),
            "raw": len(raw_rows),
        },
        "filtered_is_official_eval_split": True,
        "required_fields": list(REWARD_BENCH_REQUIRED),
        "chosen_rejected_are_strings_on_sample": all(
            isinstance(row.get("chosen"), str) and isinstance(row.get("rejected"), str) for row in sample
        ),
        "subset_counts_filtered": dict(sorted(subsets.items(), key=lambda kv: kv[0])),
        "sample_n": len(sample),
        "sample_ids": [row.get("id") for row in sample],
        "sample_rule": f"filtered split, sort by integer id, take first {REWARD_BENCH_SAMPLE_N}",
        "schema_errors": errors,
        "leaderboard_comparison": "not_run",
        "scoring": {
            "ran": False,
            "reason": (
                "No compatible small reward-model scorer is available. "
                "Qwen2.5-0.5B-Instruct is a causal LM, not an official RewardBench RM. "
                "Any logprob/proxy score would not be a RewardBench leaderboard result and was not computed."
            ),
            "proxy_label_if_any_score_had_been_computed": "proxy_not_leaderboard",
        },
        "hyper": hyper_note,
        "sample_sha256": digest,
        "sample_bytes": sample_path.stat().st_size,
        "filtered_parquet": str(filtered_parquet),
        "raw_parquet": str(raw_parquet),
    }
    write_json(out_dir / "SCHEMA_VALIDATION.json", schema)
    ids_payload = {
        "revision": REWARD_BENCH_REVISION,
        "repo_id": REWARD_BENCH_REPO,
        "split": "filtered",
        "sample_ids": [row.get("id") for row in sample],
        "n_filtered": len(filtered_rows),
        "n_raw": len(raw_rows),
    }
    write_json(out_dir / "sample_ids.json", ids_payload)
    manifest = f"""# RewardBench schema check

- **Source:** `{REWARD_BENCH_REPO}`
- **Pinned revision:** `{REWARD_BENCH_REVISION}`
- **Version:** RewardBench v1 (Lambert et al., 2024). This is **not** RewardBench-2.
- **Download source:** `{download_info["source"]}`
- **Official eval split used for schema:** `filtered` (n={len(filtered_rows)}); `raw` n={len(raw_rows)}

## Schema

Required fields observed: `prompt`, `chosen`, `rejected`, `chosen_model`, `rejected_model`, `subset`, `id`.
On this revision, `chosen` and `rejected` are strings (single-turn pairs).

Tiny committable sample: first {REWARD_BENCH_SAMPLE_N} filtered rows by integer `id`
(`schema_sample.json`, {sample_path.stat().st_size} bytes).

## What was not done

- No comparison to the published RewardBench leaderboard.
- No reward-model scoring. There is no compatible small RM scorer in this repo.
  Qwen2.5-0.5B-Instruct is not an official RewardBench reward model.
- Did not re-clone https://github.com/allenai/reward-bench. HyPER lists it as a
  submodule at `idea_A/baselines/HyPER/reward-bench` and that tree was intentionally
  left uninitialized.

Rebuild: `python scripts/download_benchmarks.py`
"""
    (out_dir / "SUBSET_MANIFEST.md").write_text(manifest, encoding="utf-8")
    loader = '''#!/usr/bin/env python3
"""Load the committed RewardBench schema sample. Does not score a leaderboard."""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_sample():
    return json.loads((HERE / "schema_sample.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    rows = load_sample()
    print(f"n={len(rows)} fields={sorted(rows[0].keys()) if rows else []}")
    print("This sample is a schema check, not a RewardBench leaderboard score.")
'''
    (out_dir / "load_sample.py").write_text(loader, encoding="utf-8")
    return {
        "status": "ok" if schema["ok"] else "schema_errors",
        "download": download_info,
        "schema": {
            "ok": schema["ok"],
            "errors": errors,
            "n_filtered": len(filtered_rows),
            "n_raw": len(raw_rows),
            "sample_n": len(sample),
            "rewardbench_version": schema["rewardbench_version"],
            "leaderboard_comparison": "not_run",
        },
        "paths": {
            "sample_json": str(sample_path),
            "manifest": str(out_dir / "SUBSET_MANIFEST.md"),
            "schema": str(out_dir / "SCHEMA_VALIDATION.json"),
        },
        "hyper": hyper_note,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-multipref", action="store_true")
    parser.add_argument("--skip-rewardbench", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("HF_HOME", str(HF_HOME))
    os.environ.setdefault("HF_HUB_CACHE", str(HF_HOME / "hub"))
    HF_HOME.mkdir(parents=True, exist_ok=True)
    DATA_CACHE.mkdir(parents=True, exist_ok=True)

    report = {
        "script": "scripts/download_benchmarks.py",
        "started_at_utc": utc_now(),
        "disk_start": disk_snapshot(ROOT),
        "python": sys.executable,
        "multipref": None,
        "rewardbench": None,
        "blockers": [],
    }
    try:
        if not args.skip_multipref:
            report["multipref"] = build_multipref(ROOT / "idea_A" / "benchmarks" / "MultiPref")
    except Exception as exc:
        traceback.print_exc()
        report["multipref"] = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
        report["blockers"].append(f"MultiPref: {type(exc).__name__}: {exc}")
    try:
        if not args.skip_rewardbench:
            report["rewardbench"] = build_rewardbench(ROOT / "idea_A" / "benchmarks" / "RewardBench")
    except Exception as exc:
        traceback.print_exc()
        report["rewardbench"] = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
        report["blockers"].append(f"RewardBench: {type(exc).__name__}: {exc}")
    report["finished_at_utc"] = utc_now()
    report["disk_end"] = disk_snapshot(ROOT)
    write_json(ROOT / "reports" / "download_benchmarks_run.json", report)
    print(json.dumps(jsonable(report), indent=2, ensure_ascii=False))
    return 1 if report["blockers"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
