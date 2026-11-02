#!/usr/bin/env python3
"""Render the source audit and compatibility table from the manifests."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    sources = yaml.safe_load((ROOT / "manifests" / "source_repositories.yaml").read_text(encoding="utf-8"))
    matrix = yaml.safe_load((ROOT / "configs" / "experiment_matrix.yaml").read_text(encoding="utf-8"))
    lines = [
        "# 官方源码审计",
        "",
        "克隆使用 HTTPS、depth 1。SHA 是当时的远端 HEAD。失败记为 `SOURCE_UNAVAILABLE`，没有用自写启发式顶替。",
        "",
        "| 名称 | 状态 | SHA | 许可证文件 | 子模块 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for repo in sources["repositories"]:
        lic = (repo.get("license") or {}).get("file") or "无"
        first = (repo.get("license") or {}).get("first_line") or ""
        if first:
            lic = f"{lic}：{first[:80]}"
        sub = (repo.get("submodules") or {}).get("status") or ""
        sha = repo.get("commit_sha") or ""
        reason = (repo.get("failure_reason") or "").replace("\n", " ")[:180]
        lines.append(
            f"| {repo['name']} | {repo['clone_status']} | `{sha}` | {lic} | {sub} |"
        )
    lines.extend(["", "## 失败与子模块", ""])
    for repo in sources["repositories"]:
        if repo["clone_status"] != "CLONED":
            reason = (repo.get("failure_reason") or "").replace("\n", " ")[:300]
            lines.append(f"- `{repo['name']}`：{reason}")
        sub = repo.get("submodules") or {}
        if sub.get("status") not in (None, "NO_SUBMODULES", "INITIALIZED"):
            detail = (sub.get("detail") or "").replace("\n", " ")[:300]
            lines.append(f"- `{repo['name']}` 子模块 {sub.get('status')}：{detail}")
    lines.extend(
        [
            "",
            "第一次克隆 HinFlow、robosuite、LIBERO 时 180 秒超时。随后 600 秒重试得到了上面的 SHA。",
            "",
            "入口只记录已经在检出里看到的文件。这些入口都没有被执行。",
            "",
            "- CUPID：`eval_demonstration_scores.py`、`train_trak_diffusion.py`、`finalize_trak.py`。README 写的是 RoboMimic 与 PushT，建在 Diffusion Policy 上。",
            "- DemInf：`scripts/quality/estimate_quality.py`。JAX/OpenX，README 要求相对动作，没有 LIBERO。",
            "- DataMIL：`scripts/perform_libero_dm.sh`。LIBERO 路径使用 Octo，另有 MetaWorld，后者不在本阶段。",
            "- Demo-SCORE：`demo_score/demo_score/train_trajwise_classifier.py`。需要成功和失败 rollout。仓库根没有 LICENSE。",
            "- STRAP：`strap/retrieval/retrieval.py` 与 `strap/embedding/encode_datasets.py`。LIBERO-10/90，子轨迹，DINOv2。策略训练在 `robomimic_strap`。仓库根没有 LICENSE。",
            "- HinFlow：`scripts/train_planner.py`、`scripts/train_hinflow_policy.py`。README 的 LIBERO 任务名是 libero_butter、libero_book、libero_chocolate、libero_microwave。仓库根没有 LICENSE。子模块超时。",
            "- Diffusion Policy：`diffusion_policy/config/train_diffusion_unet_image_workspace.yaml`。",
            "- RoboMimic、robosuite、LIBERO：基准源码。LIBERO 下载脚本在 `benchmark_scripts/download_libero_datasets.py`，本次没有运行。",
            "",
        ]
    )
    (ROOT / "docs" / "OFFICIAL_SOURCE_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    rows = [
        "# 兼容矩阵",
        "",
        "实现状态来自适配器和 `manifests/method_facts.yaml`。仓库 URL 本身不会把格子标成 READY。没有 Image-DP 训练任务被标成可跑。",
        "",
        "| 方法 | 类型 | RoboMimic | LIBERO | Image-DP | 原文复现 | 统一适配 | 在线 | 实现状态 | 排行榜 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in matrix["rows"]:
        rows.append(
            "| {method} | {kind} | {robomimic} | {libero} | {image_dp_compatibility} | {original_paper_reproduction} | {unified_adaptation} | {online_closed_loop} | {implementation_status} | {leaderboard_eligible} |".format(
                **row
            )
        )
    rows.extend(["", "## 不支持或未验证的组合", ""])
    for item in matrix["unsupported"]:
        rows.append(f"- {item.get('method')} / {item.get('benchmark')} / {item.get('metric', '')}：{item['reason']}")
    rows.extend(["", "## 当前可跑", ""])
    for job in matrix["runnable_jobs"]:
        rows.append(f"- {job['method']} on {job['benchmark']} ({job['kind']}, {job['device']})")
    rows.append("")
    (ROOT / "docs" / "COMPATIBILITY_MATRIX.md").write_text("\n".join(rows), encoding="utf-8")
    print("rendered")


if __name__ == "__main__":
    main()
