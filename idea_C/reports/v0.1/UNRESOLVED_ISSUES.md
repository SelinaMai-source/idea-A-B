# UNRESOLVED_ISSUES (v0.1)

日期：2026-11-02。`match-p1-push`。官方仓仍无。P1 overlap 已跑（A 3-seed+30-seed；B 3-seed + 30-seed except Closed）。

## 官方代码

四篇都没有可标 `OFFICIAL` 的仓库。UAI 正文写了 accompanying code，公开仓对不上，标 `INCOMPLETE`。详见 `OFFICIAL_CODE_AUDIT.md`。实现一律 `INDEPENDENT_REIMPLEMENTATION`。

## 未能在 PDF 定位的对象

- **AAAI 附录数值表**：OJS PDF 下载被 Anubis 换成 HTML（7040 字节）。`aaai.app_tables` = `NOT_LOCATED_IN_PDF`。
- **AAAI Figure 1 的 K**：HTML 摘录未印数字；config 暂用 K=50。
- **AAAI Figure 4 完整 DGP**：除笔记中的 K=10、|H1|=2 外，未能从 PDF 复核。
- **AAAI Lévy 稳定性指数 / scale**：原文未给。代码 1.5 / 1.0，`unspecified_in_paper=True`。
- **SCORE Appendix E 的 T**：暂按正文合成 T=1000。

## 数据

- **e-PS Figure 4 OPT-1.3B** 预生成水印流未发布 → `BLOCKED_NO_DATA`。禁止用其他模型代替。
- **Jester**：下列 URL 本轮失败 → `BLOCKED_NO_DATA`。
  - `https://eigentaste.berkeley.edu/dataset/jester_dataset_2.zip` TLS `SSL_ERROR_SYSCALL`
  - `https://eigentaste.berkeley.edu/dataset/archive/jester_dataset_2.zip` TLS fail
  - `https://goldberg.berkeley.edu/jester-data/jester_dataset_2.zip` HTTP 404
  - `https://www.kaggle.com/datasets/vikashrajluhaniwal/jester-17m-jokes-ratings-dataset` curl HTTP 404
- **SCORE Yeast**：`http://chemgrid.org/cgm/` HTTP 410 Gone。Nature 2016.95 是 OD600，不是 417026 p 值流。`BLOCKED_NO_DATA`。
- **SCORE 心梗**：`https://figshare.le.ac.uk/articles/dataset/Myocardial_infarction_complications_Database/12045261` WAF challenge (HTTP 202)。`BLOCKED_NO_DATA`。
- **UAI NYC taxi**：公开 CSV 已下载到 `idea_C/results/v0.1/data_probes/nyc_taxi.csv`（10320 行）+ NAB windows。STL / e-GAI runner 未接线，格子仍 `NOT_RUN`（不是 `BLOCKED_NO_DATA`）。

## 算力 / resume

### UAI Closed m=3000 n=80 — `BLOCKED_COMPUTE`

实测 1-seed Closed e-LOND：m=200 为 0.425s，m=400 为 3.929s。立方外推 m=3000 单 seed ≈1435s，n=80 ≈ **31.89h** > 8h。未砍 closure 搜索。donation/e-LOND/r-LOND 已在 m=3000 n=80 跑完。

```bash
cd /Users/grantwright/Desktop/科研/10.30新课题/v0
export PYTHONPATH="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
nice -n 10 .venv/bin/python idea_C/scripts/run_v0_1.py --object uai.fig2_runtime --scale paper --paper-scale --run --resume
```

### e-PS 1D paper-scale 500×H=50000 — `BLOCKED_COMPUTE`

1-seed 实测：gaussian eps/oracle H=50000 = 44.562s（TPR=1.0，SAT=28253）；student_t = 51.641s（TPR=0.833，SAT=CENSORED）；rss≈664MB。500-seed 外推 ≈80h。**不要把 30-seed 标成 paper-scale。** P1 不跑 e-PS 500。

```bash
cd /Users/grantwright/Desktop/科研/10.30新课题/v0
export PYTHONPATH="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
nice -n 10 .venv/bin/python idea_C/scripts/run_v0_1.py --object eps.fig2a --scale paper --paper-scale --run --resume
nice -n 10 .venv/bin/python idea_C/scripts/run_v0_1.py --object eps.fig2c --scale paper --paper-scale --run --resume
```

### P1-B Closed 30-seed — skipped

3-seed Closed e-LOND T=1000 wall=192.656s；30-seed 外推 ~32 min（tens of minutes）。已写 `C-v0.1_RERUN_ob_closed_elond_pilot30_skipped`。其余方法 30-seed 已跑。

```bash
cd /Users/grantwright/Desktop/科研/10.30新课题/v0
export PYTHONPATH="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
nice -n 10 .venv/bin/python idea_C/scripts/run_p1b.py --v01 --scale pilot --run --assume-cpqd
```

### AAAI Gaussian / Heavy-tail paper 80000×500 — 已完成

- Fig1 Uniform SAT=6490.298 TPR=1 FDR=0；E-process 同 SAT、post-burn-in 57400；Mean-based SAT=CENSORED success=0.226
- Fig3 Uniform SAT=724.272 FDR=0.004；algo3 SAT=9487.286 FDR=0 post-burn-in
- Fig2 / Fig4 `NOT_RUN`。Fig4 无 resume，除非先接线 Algorithm 4。

### SCORE AR / Appendix

AR exponential 与 Appendix E/F runner 未接线。无 resume 命令可跑，除非先接线 DGP。

## P1

已执行。前提：AAAI Fig1/Fig3 paper-scale `protocol=ok`；SCORE 独立 Gaussian 500 与 Table 2 α=0.05 对齐。e-PS paper-scale 仍不可信，未跑 e-PS 500，P1-A 中 e-PS 标 `ADAPTED_PROTOCOL`（μ=1 共同协议）。污染非鲁棒 e-PS/Greedy 仍 `NOT_COMPARABLE`。
