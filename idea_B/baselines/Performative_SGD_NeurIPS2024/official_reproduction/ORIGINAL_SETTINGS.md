# Official synthetic-iter settings (read from upstream, not rewritten)

Source files: `synthetic-iter/run.sh`, `synthetic-iter/main.py`, `synthetic-iter/dataset.py`, `synthetic-iter/model.py`, `synthetic-iter/util.py`, `readme.md`.
Upstream commit: `ccbeeddfbf9237aa06dca3a1579529faabd8fe99`.

## Official launch command

```bash
cd synthetic-iter
mpiexec -np 10 python main.py
```

Dependencies declared by the README: `mpi4py, torch, numpy, matplotlib, pandas, sklearn, scipy`.

## Algorithm in `simu1()` (the `__main__` entry)

| Setting | Value in `main.py` |
|---|---|
| `eps_list` | `[0, 0.1, 10, 100]` |
| device | `"cuda" if torch.cuda.is_available() else "cpu"` |
| `lambda_reg` | `1e-3` |
| `c` | `0.1` |
| `a0`, `a1` | `50`, `10` (TimeVaryingSGD class exists but is unused in this entry) |
| `lr` | `1e-2` (passed in, unused by `SetOpt_CstStep_SGD`) |
| `grd_type` | `'sgd'` |
| `batch` | `5` |
| `log_maxiter` | `5` → `num_iter = 10**5` |
| model | `LinearLayerWithBias(num_input=10, num_output=1, c=0.1)` |
| optimizer | `torch.optim.SGD(..., lr=a0/sqrt(num_iter))` = `50/sqrt(1e5)` |
| performative shift | `x_new = x - eps * theta * 1{y==1}` |
| data | `load_data` **ignores** the `file_loc='circle_data.csv'` argument and always reads `data.csv` |
| label flip | `flip_flag=True`, `flip_ratio=0.1` on train split |
| MPI | `MPI.COMM_WORLD.Get_rank()` used for save filename `eps{eps}-batch{batch}-res{rank}.npy` and rank-0 prints. There is **no** gradient allreduce. Ten ranks are ten independent replicas. |

## Machine blockers for the official full run

- `mpi4py` is not installed in the project venv.
- `mpicc` / `mpiexec` are not on `PATH` (Homebrew `open-mpi` prefix exists but has no `bin/`).
- Official `mpiexec -np 10 python main.py` is therefore **BLOCKED**.
- `torch` is not in the main project venv; an isolated venv is used if a single-replica run is attempted.
- This is **not** claimed as a full paper reproduction even if a single replica completes.

Author-provided `.npy` files already exist under `synthetic-iter/output-dimi/` and `synthetic-iter/output-cnt/`. Those are **upstream artifacts**, not local runs.
