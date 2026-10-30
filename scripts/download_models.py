#!/usr/bin/env python3
"""Download Qwen2.5-0.5B-Instruct into the shared cache and run a short smoke test.

Weights live in gitignored `.cache/models/`. Idea A/B model folders only get
manifests plus relative symlinks.

Never substitutes a different architecture. Pins a commit, not floating `main`.
Does not treat this smoke test as LLM post-training.

    .isolated_baselines_venv/bin/python scripts/download_models.py
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
HF_HOME = ROOT / ".cache" / "hf"
SHARED_MODELS = ROOT / ".cache" / "models"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
PINNED_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
EXPECTED_ARCH = "Qwen2ForCausalLM"
EXPECTED_MODEL_TYPE = "qwen2"
ESTIMATED_BYTES = 1_250_000_000
MIN_FREE_AFTER_BYTES = 4 * 1024**3
SMOKE_PROMPT = "Reply with the single word pong."
MAX_NEW_TOKENS = 8

ALLOW_PATTERNS = [
    ".gitattributes",
    "LICENSE",
    "README.md",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
]


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


def relative_symlink(link: Path, target: Path) -> str:
    link.parent.mkdir(parents=True, exist_ok=True)
    rel = os.path.relpath(target, start=link.parent)
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(rel)
    return rel


def verify_config(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    arch = config.get("architectures") or []
    model_type = config.get("model_type")
    ok = EXPECTED_MODEL_TYPE == model_type and EXPECTED_ARCH in arch
    if not ok:
        raise RuntimeError(
            f"Architecture mismatch: model_type={model_type!r} architectures={arch!r}. "
            f"Expected model_type={EXPECTED_MODEL_TYPE!r} and {EXPECTED_ARCH}. "
            "Refusing to substitute another model."
        )
    return {
        "ok": True,
        "model_type": model_type,
        "architectures": arch,
        "hidden_size": config.get("hidden_size"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "vocab_size": config.get("vocab_size"),
        "torch_dtype": config.get("torch_dtype"),
    }


def download_qwen(local_dir: Path) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    before = disk_snapshot(ROOT)
    projected = before["free_bytes"] - ESTIMATED_BYTES
    if projected < MIN_FREE_AFTER_BYTES:
        return {
            "status": "BLOCKED",
            "reason": (
                f"Projected free disk after ~{ESTIMATED_BYTES} byte download would be "
                f"{projected} bytes (< {MIN_FREE_AFTER_BYTES}). Skipping Qwen2.5-0.5B-Instruct. "
                "No substitute model was downloaded."
            ),
            "disk_before": before,
            "estimated_bytes": ESTIMATED_BYTES,
            "revision": PINNED_REVISION,
        }
    local_dir.mkdir(parents=True, exist_ok=True)
    source = "huggingface"
    hf_error = None
    try:
        resolved = snapshot_download(
            repo_id=MODEL_ID,
            repo_type="model",
            revision=PINNED_REVISION,
            local_dir=str(local_dir),
            allow_patterns=ALLOW_PATTERNS,
        )
    except Exception as exc:
        hf_error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        try:
            from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot

            resolved = ms_snapshot(
                MODEL_ID,
                revision=PINNED_REVISION,
                cache_dir=str(local_dir),
                allow_patterns=ALLOW_PATTERNS,
            )
            source = "modelscope"
        except Exception as exc2:
            after = disk_snapshot(ROOT)
            return {
                "status": "failed",
                "hf_error": hf_error,
                "modelscope_error": f"{type(exc2).__name__}: {exc2}",
                "disk_before": before,
                "disk_after": after,
                "revision": PINNED_REVISION,
                "note": "Did not substitute another architecture.",
            }
    after = disk_snapshot(ROOT)
    config_check = verify_config(local_dir / "config.json")
    tokenizer_cfg = json.loads((local_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    files = official_weight_files(local_dir)
    return {
        "status": "ok",
        "source": source,
        "hf_error": hf_error,
        "repo_id": MODEL_ID,
        "revision": PINNED_REVISION,
        "resolved_path": str(resolved),
        "local_dir": str(local_dir),
        "config_check": config_check,
        "tokenizer_model_max_length": tokenizer_cfg.get("model_max_length"),
        "files": files,
        "disk_before": before,
        "disk_after": after,
        "weights_bytes": sum(item["bytes"] for item in files),
    }


def official_weight_files(local_dir: Path) -> list[dict[str, Any]]:
    files = []
    for name in ALLOW_PATTERNS:
        path = local_dir / name
        if path.is_file():
            files.append({"relative": name, "bytes": path.stat().st_size})
    return files


def link_model_folder(model_dir: Path, cache_dir: Path, idea: str, download_info: dict[str, Any]) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    linked = []
    for item in download_info.get("files") or official_weight_files(cache_dir):
        rel = item["relative"]
        if rel.startswith(".cache/") or "/" in rel:
            continue
        target = cache_dir / rel
        dest = model_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        linked.append({"name": rel, "symlink": relative_symlink(dest, target), "bytes": item["bytes"]})
    metadata = {
        "idea": idea,
        "model_id": MODEL_ID,
        "revision": PINNED_REVISION,
        "downloaded": download_info.get("status") == "ok",
        "source": download_info.get("source"),
        "shared_cache": str(cache_dir),
        "shared_cache_relative": os.path.relpath(cache_dir, start=model_dir),
        "files": linked,
        "config_check": download_info.get("config_check"),
        "recorded_at_utc": utc_now(),
        "note": "Weights are not copied. This folder holds manifests and relative symlinks.",
    }
    write_json(model_dir / "download_metadata.json", metadata)
    files_md = "\n".join(
        f"- `{item['name']}` -> `{item['symlink']}` ({item['bytes']} bytes)" for item in linked
    )
    manifest = f"""# Qwen2.5-0.5B-Instruct

- **Official repo:** `{MODEL_ID}`
- **Pinned revision (not floating main):** `{PINNED_REVISION}`
- **Source:** `{download_info.get("source")}`
- **Architecture check:** `{json.dumps(download_info.get("config_check"))}`
- **Shared cache (gitignored):** `{cache_dir}`
- **This folder:** manifests + relative symlinks only

{files_md}

This download is a feasibility / tokenizer+weights load check. It is **not**
LLM post-training and must not be reported as Idea A/B synthetic-matrix evidence.
"""
    (model_dir / "MODEL_MANIFEST.md").write_text(manifest, encoding="utf-8")
    write_json(model_dir / "manifest.json", metadata)


def rss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports ru_maxrss in bytes; Linux in kilobytes.
    if sys.platform == "darwin":
        return int(usage)
    return int(usage) * 1024


def _dtype_from_name(torch_mod: Any, name: str) -> Any:
    mapping = {
        "float16": torch_mod.float16,
        "float32": torch_mod.float32,
        "bfloat16": getattr(torch_mod, "bfloat16", torch_mod.float16),
    }
    return mapping[name]


def inference_worker(model_dir: Path, device: str, dtype_name: str) -> dict[str, Any]:
    """In-process worker. MPS generate may abort the process; call via subprocess."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    result: dict[str, Any] = {
        "status": "failed",
        "model_id": MODEL_ID,
        "revision": PINNED_REVISION,
        "prompt": SMOKE_PROMPT,
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "device": device,
        "dtype_name": dtype_name,
        "torch_version": torch.__version__,
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "mps_built": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_built()),
        "python": sys.executable,
        "is_llm_post_training": False,
        "is_synthetic_matrix": False,
    }
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=False)
    result["tokenizer_load_ok"] = True
    result["tokenizer_load_seconds"] = time.perf_counter() - t0

    dtype = _dtype_from_name(torch, dtype_name)
    t1 = time.perf_counter()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), dtype=dtype, trust_remote_code=False
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), torch_dtype=dtype, trust_remote_code=False
        )
    model.to(device)
    model.eval()
    result["weights_load_ok"] = True
    result["weights_load_seconds"] = time.perf_counter() - t1
    result["dtype"] = str(dtype)

    messages = [{"role": "user", "content": SMOKE_PROMPT}]
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = SMOKE_PROMPT
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    if device == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    t2 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    result["generate_ok"] = True
    result["generate_latency_seconds"] = time.perf_counter() - t2
    result["output_text"] = tokenizer.decode(out[0], skip_special_tokens=True)
    result["n_output_tokens"] = int(out.shape[-1])
    peak = {"ru_maxrss_bytes": rss_bytes()}
    if device == "mps":
        try:
            peak["torch_mps_current_allocated_bytes"] = int(torch.mps.current_allocated_memory())
            peak["torch_mps_driver_allocated_bytes"] = int(torch.mps.driver_allocated_memory())
        except Exception as exc:
            peak["torch_mps_error"] = f"{type(exc).__name__}: {exc}"
    result["peak_memory"] = peak
    result["status"] = "ok"
    result["clean_exit"] = True
    return result


def run_inference_smoke(model_dir: Path) -> dict[str, Any]:
    """Prefer MPS, but isolate generate in a child process.

    An in-process MPS generate on this Mac aborted with an ObjC NSException that
    Python cannot catch. The child process lets us record that crash and then
    retry CPU without swapping models.
    """
    summary: dict[str, Any] = {
        "status": "failed",
        "model_id": MODEL_ID,
        "revision": PINNED_REVISION,
        "prompt": SMOKE_PROMPT,
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "is_llm_post_training": False,
        "is_synthetic_matrix": False,
        "python": sys.executable,
        "note": "Short deterministic load/generate check only. Not post-training.",
        "attempts": [],
    }
    try:
        import torch

        mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        summary["torch_version"] = torch.__version__
        summary["mps_available"] = mps_available
        summary["mps_built"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_built())
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["hint"] = "Use .isolated_baselines_venv/bin/python which has torch and transformers."
        return summary

    plan: list[tuple[str, str]] = []
    if mps_available:
        plan.append(("mps", "float16"))
        plan.append(("mps", "float32"))
    plan.append(("cpu", "float32"))

    successful = None
    for device, dtype_name in plan:
        handle, tmp_name = tempfile.mkstemp(prefix="qwen_smoke_", suffix=".json")
        os.close(handle)
        out_path = Path(tmp_name)
        proc = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--infer-worker",
                "--model-dir",
                str(model_dir),
                "--device",
                device,
                "--dtype",
                dtype_name,
                "--output-json",
                str(out_path),
            ],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        attempt: dict[str, Any] = {
            "device": device,
            "dtype_name": dtype_name,
            "returncode": proc.returncode,
        }
        stderr = proc.stderr or ""
        if out_path.exists() and out_path.stat().st_size:
            try:
                attempt["worker"] = json.loads(out_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                attempt["output_json_raw"] = out_path.read_text(encoding="utf-8")[-4000:]
        out_path.unlink(missing_ok=True)
        if proc.returncode != 0:
            attempt["status"] = "crashed" if proc.returncode < 0 else "failed"
            attempt["stderr_head"] = stderr[:1500]
            attempt["stderr_tail"] = stderr[-1500:]
            summary["attempts"].append(attempt)
            continue
        worker = attempt.get("worker") or {}
        if worker.get("status") == "ok":
            attempt["status"] = "ok"
            summary["attempts"].append(attempt)
            successful = worker
            break
        attempt["status"] = worker.get("status") or "failed"
        if stderr:
            attempt["stderr_tail"] = stderr[-4000:]
        summary["attempts"].append(attempt)

    if successful is None:
        summary["error"] = "All device/dtype generate attempts failed. See attempts."
        return summary

    summary.update(
        {
            "status": "ok",
            "device": successful.get("device"),
            "dtype": successful.get("dtype"),
            "tokenizer_load_ok": successful.get("tokenizer_load_ok"),
            "tokenizer_load_seconds": successful.get("tokenizer_load_seconds"),
            "weights_load_ok": successful.get("weights_load_ok"),
            "weights_load_seconds": successful.get("weights_load_seconds"),
            "generate_ok": successful.get("generate_ok"),
            "generate_latency_seconds": successful.get("generate_latency_seconds"),
            "output_text": successful.get("output_text"),
            "n_output_tokens": successful.get("n_output_tokens"),
            "peak_memory": successful.get("peak_memory"),
            "clean_exit": True,
            "preferred_device": "mps" if mps_available else "cpu",
            "device_actually_used": successful.get("device"),
        }
    )
    if successful.get("device") != "mps" and mps_available:
        summary["mps_generate_blocker"] = (
            "MPS was available and was tried first. generate() aborted or failed "
            "on MPS; CPU was used as a fallback on the same Qwen2.5-0.5B-Instruct "
            "weights. No other model was substituted."
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--infer-worker", action="store_true")
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", dest="dtype_name", default="float32")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()
    if args.infer_worker:
        model_dir = args.model_dir or (SHARED_MODELS / "Qwen2.5-0.5B-Instruct")
        try:
            payload = inference_worker(model_dir, args.device, args.dtype_name)
            rc = 0 if payload.get("status") == "ok" else 1
        except Exception as exc:
            payload = {
                "status": "failed",
                "device": args.device,
                "dtype_name": args.dtype_name,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "is_llm_post_training": False,
            }
            rc = 1
        text = json.dumps(jsonable(payload), ensure_ascii=False)
        if args.output_json:
            args.output_json.write_text(text + "\n", encoding="utf-8")
        else:
            print(text)
        return rc
    os.environ.setdefault("HF_HOME", str(HF_HOME))
    os.environ.setdefault("HF_HUB_CACHE", str(HF_HOME / "hub"))
    HF_HOME.mkdir(parents=True, exist_ok=True)
    SHARED_MODELS.mkdir(parents=True, exist_ok=True)

    cache_dir = SHARED_MODELS / "Qwen2.5-0.5B-Instruct"
    idea_a = ROOT / "idea_A" / "models" / "Qwen2.5-0.5B-Instruct"
    idea_b = ROOT / "idea_B" / "models" / "Qwen2.5-0.5B-Instruct"

    report: dict[str, Any] = {
        "script": "scripts/download_models.py",
        "started_at_utc": utc_now(),
        "disk_start": disk_snapshot(ROOT),
        "python": sys.executable,
        "model_id": MODEL_ID,
        "revision": PINNED_REVISION,
        "download": None,
        "inference": None,
        "blockers": [],
    }
    if args.skip_download:
        download_info = {
            "status": "ok" if (cache_dir / "model.safetensors").exists() else "missing",
            "source": "huggingface",
            "cache_reuse": True,
            "local_dir": str(cache_dir),
            "files": official_weight_files(cache_dir),
            "weights_bytes": sum(item["bytes"] for item in official_weight_files(cache_dir)),
        }
        if (cache_dir / "config.json").exists():
            download_info["config_check"] = verify_config(cache_dir / "config.json")
        report["download"] = download_info
    else:
        report["download"] = download_qwen(cache_dir)

    download_info = report["download"]
    if download_info.get("status") != "ok":
        report["blockers"].append(download_info.get("reason") or download_info.get("hf_error") or download_info)
        write_json(idea_a / "download_metadata.json", download_info)
        write_json(idea_b / "download_metadata.json", download_info)
        blocked_manifest = (
            f"# Qwen2.5-0.5B-Instruct\n\nStatus: `{download_info.get('status')}`\n\n"
            f"{json.dumps(jsonable(download_info), indent=2)}\n"
        )
        (idea_a / "MODEL_MANIFEST.md").write_text(blocked_manifest, encoding="utf-8")
        (idea_b / "MODEL_MANIFEST.md").write_text(blocked_manifest, encoding="utf-8")
        if download_info.get("status") == "BLOCKED":
            write_json(
                idea_a / "INFERENCE_SMOKE.json",
                {"status": "BLOCKED", "reason": download_info.get("reason"), "is_llm_post_training": False},
            )
        report["finished_at_utc"] = utc_now()
        report["disk_end"] = disk_snapshot(ROOT)
        write_json(ROOT / "reports" / "download_models_run.json", report)
        print(json.dumps(jsonable(report), indent=2, ensure_ascii=False))
        return 1

    link_model_folder(idea_a, cache_dir, "A", download_info)
    link_model_folder(idea_b, cache_dir, "B", download_info)

    if args.skip_inference:
        smoke = {"status": "skipped", "is_llm_post_training": False}
    else:
        smoke = run_inference_smoke(cache_dir)
    write_json(idea_a / "INFERENCE_SMOKE.json", smoke)
    write_json(idea_b / "INFERENCE_SMOKE.json", smoke)
    report["inference"] = {
        "status": smoke.get("status"),
        "device": smoke.get("device"),
        "generate_ok": smoke.get("generate_ok"),
        "generate_latency_seconds": smoke.get("generate_latency_seconds"),
        "peak_memory": smoke.get("peak_memory"),
        "error": smoke.get("error"),
    }
    if smoke.get("status") != "ok" and not args.skip_inference:
        report["blockers"].append(smoke.get("error") or smoke)
    report["finished_at_utc"] = utc_now()
    report["disk_end"] = disk_snapshot(ROOT)
    write_json(ROOT / "reports" / "download_models_run.json", report)
    print(json.dumps(jsonable(report), indent=2, ensure_ascii=False))
    return 0 if not report["blockers"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
