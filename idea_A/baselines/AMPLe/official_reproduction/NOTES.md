# AMPLe official run notes

Upstream commit: `0cad341f6a9e77063ef451c27732bf5c1534c797`.

## Official script

`main.py` is the preference-learning experiment. Acquisition options used in the README are `volume` and `random`. This local attempt keeps **`acquisition=volume`** and **`burnin=50000`**. Only `n_rounds` is shortened (README example uses 100; argparse default is 10). Label: **PARTIAL**.

## Why `--dims 0 1` was omitted

The README assistant example includes `--dims 0 1`. In `main.py` that rewrite happens **before** `load_dataset`, so the requested Hub id becomes `minhyeonoh/feedback_assistant_harmlesshelpful_271`. Probed on 2026-10-30: HTTP 401 / not public.

The public dataset that matches `--rm_names harmless helpful humor` without `--dims` is:

- `minhyeonoh/feedback_assistant_harmlesshelpfulhumor_271` (download_size 4,531,806 bytes, 10,240 train examples)

Omitting `--dims` is required for a faithful load of released data. It does **not** change the acquisition rule.

## Isolation

Torch / datasets / accelerate / wandb are installed in `/Users/grantwright/Research/idea-A-B/.isolated_baselines_venv`, not the main project venv. Forbidden packages (vLLM, bitsandbytes, flash-attn, CUDA, NCCL, deepspeed) are not installed. `rewardsincontext/requirements.txt` is **not** used (it contains bitsandbytes and flash-attn).

`WANDB_MODE=offline` is required because `wandb.init(entity="personalization", project="ACL25")` is unconditional in `main.py`.
