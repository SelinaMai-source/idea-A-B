# 共享权重缓存

实际权重复用一份，放在仓库根的 gitignored 目录：

`../../.cache/models/`

（即 `/Users/grantwright/Research/idea-A-B/.cache/models/`）

Idea A 与 Idea B 的 `models/Qwen2.5-0.5B-Instruct/` 只放 `MODEL_MANIFEST.md`、`download_metadata.json` 和指向该共享缓存的相对符号链接，避免复制两份。

本目录本身不再存放权重 blob。下载与烟测由 `scripts/download_models.py` 完成。
