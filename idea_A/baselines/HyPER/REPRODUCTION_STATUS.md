# HyPER official reproduction status

**Status: BLOCKED (pre-declared).** No simplified HyPER was implemented.

- **Upstream:** https://github.com/allenai/hybrid-preferences
- **Commit:** `4d96b27b989828b0b69a430f68d25b6b36e9015a`
- **Clone:** `git clone --depth 1 --recurse-submodules=no` (submodules not initialized)
- **On-disk size:** 632K (clone not aborted; well under the disk budget)

## What the official pipeline requires

Read from `README.md`, `docs/01-training-proxy-reward-models.md`, `docs/02-evaluating-proxy-reward-models.md`, `docs/03-fitting-a-regressor-and-getting-best-mix.md`, `scripts/submit_tpu_train_job.py`, `requirements.txt`, `.gitmodules`.

1. **TPU / GCP (hard blocker on this MacBook Air).** Docs instruct creating a Google Cloud TPU (`TPU_TYPE=v3-128`, `TPU_ZONE=us-east1-d`, `GCP_PROJECT=ai2-tpu`) and submitting EasyLM RM / DPO jobs with `--load_llama_config='13b'` or Llama-3.1-8B. `scripts/create_tpu_single.sh`, `scripts/submit_tpu_train_job.py`, `scripts/stop_tpu_jobs.sh` are TPU-only.
2. **vLLM (forbidden on this machine).** `requirements.txt` pins `vllm>=0.4.1`. Local policy: do not install vLLM / bitsandbytes / flash-attn / CUDA / NCCL.
3. **Reward-model training at 8B–13B (hard blocker).** Official RM training uses EasyLM on TPU with Tulu-2 13B / Llama-3.1-8B checkpoints stored on GCS (`ljm-dev` bucket paths in docs). This laptop has 16GB RAM, no TPU, no CUDA, ~12GB free disk.
4. **Beaker / AI2 cluster (hard blocker).** Evaluation conversion jobs (`evals/generate_eval_runs.py`, `evals/convert_to_hf.py`) launch Beaker experiments and fetch RewardBench scores via `scripts/fetch_evals_rewardbench.py`. Makefile target `create-beaker` pushes an AI2 Beaker image.
5. **Submodules not cloned.** `.gitmodules` lists `open-instruct` and `reward-bench`. They were intentionally not initialized. Empty placeholder directories exist from the parent clone.
6. **Additional Linux-oriented deps in `requirements.txt`:** `beaker-py`, `google-cloud-storage`, `sentence-transformers`, `spacy==3.7.5`, `bert_score`. Installing them would not remove the TPU/vLLM/RM blockers.

## What was *not* done

- No invented “simplified HyPER”.
- No dummy routing policy, no synthetic PPM, no fake RewardBench numbers.
- No submodule init, no vLLM install, no model download.

## Comparability

HyPER optimizes human vs. LM annotation routing for downstream RM / RewardBench scores. That objective, budget, and information are **NOT DIRECTLY COMPARABLE** to Idea A’s Neyman-audit IPW protocol.
