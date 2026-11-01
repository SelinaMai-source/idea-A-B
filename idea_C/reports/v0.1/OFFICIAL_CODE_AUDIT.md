# Idea C v0.1 官方代码审计

审计日期：2026-11-02。工作副本 `/Users/grantwright/Desktop/科研/10.30新课题/v0`。分支 `idea-c-v0.1` 由本地 `idea-c-v0` HEAD `f5062fe5438d3df37e876b5fa5f13c68c37edab4` 新建并检出。该 HEAD 与 `origin/idea-c-v0` 相同（`git rev-list --left-right --count` 为 `0 0`）。未 reset、未 rebase、未改 git config、未 commit、未 push、未 `git add`。Idea A/B 脏文件与 `idea_B/results/v1.1/**/plots/` 保持未暂存。

本文件只记录公开来源里有没有**这篇论文**的官方代码。未实现算法，未改 burn-in，未跑 Monte Carlo，未设计 C-v1。UAI 2026 **不是** CCF-A Main。e-PS **是预印本**，不是已发表 CCF-A。

未找到可标 `OFFICIAL` 的仓库，因此本轮没有新的浅克隆，也没有新增 MANIFEST。磁盘上已有的两棵树仍是错配相关代码，见第 6 节。

## 总表

| Paper | Official Repo | Evidence | Commit | Environment | Data | Figure Scripts | Missing Artifacts | Implementation Mode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| On the Robustness of Bandit Multiple Testing (Zhengyu Zhou, Weiwei Liu), AAAI 2026 | 无 | 出版社页与 PDF 下载被 Anubis 换成 HTML，未读到 PDF 字节。ACM 摘要、作者主页、GitHub 仓库检索均无本文代码链接。见 §1 | 无 | 无官方环境文件 | 正文实验为仿真；本轮未读到 PDF，不能从 PDF 再列数据文件 | 未定位 | 官方仓库；本轮未能打开的 AAAI PDF/附录 | `INDEPENDENT_REIMPLEMENTATION` + `NOT_FOUND` |
| SCORE: A Unified Framework for Overshoot Refund in Online FDR Control (Qi Kuang, Bowen Gang, Yin Xia), ICML 2026 / arXiv:2601.20386 | 无 | PMLR 页只有 PDF。arXiv 源码无 GitHub URL。OpenReview 有补充材料 zip，但匿名下载 403，**没有打开 zip**。见 §2 | 无 | 无 | 心梗数据有 figshare 链接；合成数据由论文 DGP 生成 | 未定位 | 官方仓库；未打开的 OpenReview zip | `INDEPENDENT_REIMPLEMENTATION` + `NOT_FOUND` |
| Improving online FDR procedures via online analogs of e-closure and compound e-values (Ziyu Xu, Lasse Fischer, Aaditya Ramdas), UAI 2026 / arXiv:2603.24792。**非 CCF-A Main** | 无本文仓库。正文写 “accompanying code”，公开仓对不上 | arXiv 源码有这句话，没有 URL。作者主页列出本文但不挂 code。Xu/Fischer 公开仓已逐个看过。见 §3 | 无本文 commit。错配仓 SHA 见 §6 | 无本文环境文件 | NYC taxi 等出现在离线 e-closure 仓，不是本 UAI 发布 | 未定位本文脚本 | 正文声称的 accompanying code | `INDEPENDENT_REIMPLEMENTATION` + `INCOMPLETE` |
| Sample-Efficient Multiple Testing with Adaptive Data Collection / e-PS (Zhanran Lin, Wanteng Ma, Zhimei Ren, Yuting Wei), arXiv:2609.26651 **预印本** | 无 | arXiv 摘要页与源码无代码 URL。作者主页列出本文但不挂 code。见 §4 | 无 | 无 | Jester 评分在 Kaggle 数据集页公开列出。OPT-1.3B 预生成水印流未找到发布 | 未定位 | 官方代码；作者预生成的 OPT-1.3B 水印流 | `INDEPENDENT_REIMPLEMENTATION` + `NOT_FOUND` |

`OFFICIAL` 只在仓库就是这篇论文时使用。上表四行都不是。

## 检索范围与限制

对每篇都查了：出版社 / PMLR / arXiv abs、HTML、源码包；OpenReview API；作者主页；GitHub 仓库检索（全名、算法名、arXiv 号）和作者公开仓列表；Zenodo；Software Heritage；`idea_C/reports/v0/BASELINE_AUDIT.md` 与 `idea_C/baselines/eclosure/` 的两份 MANIFEST。

限制，写在对应结论旁边：

- GitHub **code search** API 未登录返回 401。仓库检索和作者公开仓枚举是成功的。
- Software Heritage 搜索页返回 Anubis 人机验证，本轮没有完成 SWH 查询。四篇正文都没有给出 SWH 标识。
- Zenodo 按 arXiv 号 `2601.20386`、`2603.24792`、`2609.26651` 查询的 `hits.total` 为 0。随后按标题精确查询返回 403，没有再确认。
- OpenReview 附件匿名下载返回 `ChallengeRequiredError`（HTTP 403）。附件**存在**，但本轮没有读到字节。

## 1. AAAI 2026 — On the Robustness of Bandit Multiple Testing

作者：Zhengyu Zhou, Weiwei Liu（武汉大学）。

| 来源 | URL | 本轮看到的内容 |
| --- | --- | --- |
| AAAI OJS | https://ojs.aaai.org/index.php/AAAI/article/view/40148 | WebFetch 与 curl 都停在 Anubis “Making sure you're not a bot”。Related Material 区块没有读到 |
| AAAI PDF | https://ojs.aaai.org/index.php/AAAI/article/download/40148/44109 | HTTP 200，但正文是 7040 字节 HTML，不是 PDF。附录是否另附，本轮不能从 PDF 判断 |
| ACM | https://dl.acm.org/doi/10.1609/aaai.v40i34.40148 | 检索摘要只有摘要与作者，没有代码或补充材料链接 |
| researchr | https://researchr.org/publication/ZhouL26a | 页码 29107–29114。摘要缺失，无代码链接 |
| arXiv | 未找到同名预印本 | 检索落到 AAAI/ACM，没有 abs 页 |
| OpenReview | API `notes/search?term=On the Robustness of Bandit Multiple Testing Zhou` | 前几条不是这篇。没有本文 forum、PDF 或附件 |
| 作者主页 | https://sites.google.com/site/weiweiliuhomepage/ | 第 6 条列出本文（AAAI 2026, Oral），条目本身没有 code / github。邮箱：`liuweiwei863@gmail.com` |
| ACM 作者页 | https://dl.acm.org/profile/99661452650 | Zhengyu Zhou，武汉大学。无代码链接 |
| GitHub 仓库检索 | API `search/repositories?q=Robustness+Bandit+Multiple+Testing` | `total_count=0` |
| GitHub 用户 | `zhouliu` https://github.com/zhouliu | 公开仓是 ICLR 2019 图像复原 `RNAN` 与测试主页，不是本文 |
| Zenodo / SWH | arXiv 号不适用；SWH 页被人机验证拦住 | 无本文记录可引用 |

**结论：`NOT_FOUND`。** 实现模式 `INDEPENDENT_REIMPLEMENTATION`。

不要当成官方代码的仓库：

- https://github.com/neilzxu/e_bmt — NeurIPS 2021 *A unified framework for bandit multiple testing*（Xu, Wang, Ramdas）。API：默认分支 `master`，homepage https://arxiv.org/abs/2107.07322 ，最后 push `2022-03-09T01:37:52Z`，license 字段为空。不是 Zhou & Liu 2026。

## 2. ICML 2026 — SCORE

全名：SCORE: A Unified Framework for Overshoot Refund in Online FDR Control。Qi Kuang, Bowen Gang, Yin Xia。复旦大学统计与数据科学系。PMLR v306，arXiv:2601.20386（检索到的版本信息含 v2，2026-05-29）。

| 来源 | URL | 本轮看到的内容 |
| --- | --- | --- |
| PMLR | https://proceedings.mlr.press/v306/kuang26c.html | 只有 PDF：https://raw.githubusercontent.com/mlresearch/v306/main/assets/kuang26c/kuang26c.pdf 。页内无 Software、无补充材料、无 GitHub |
| 猜测的 PMLR 补充文件 | https://raw.githubusercontent.com/mlresearch/v306/main/assets/kuang26c/kuang26c-supp.pdf | HTTP 404 |
| ICML 海报页 | https://icml.cc/virtual/2026/poster/61407 | 摘要与海报信息。检索摘录无代码链接 |
| arXiv abs / HTML | https://arxiv.org/abs/2601.20386 | 无 code 栏。通讯作者 `xiayin@fudan.edu.cn`（abs 页与源码 `icml2026_cr.tex` 的 `\icmlcorrespondingauthor` 一致） |
| arXiv 源码 | https://arxiv.org/e-print/2601.20386 （gzip，2026-06-01） | 解包后 `.tex` / `.bib` 无 `github`。唯一的数据 URL 是心梗数据库 figshare。源码包内是正文 tex 与 `figure/*.pdf`，没有作图脚本 |
| OpenReview | https://openreview.net/forum?id=qX4Nm7eNM5 | API 笔记 `qX4Nm7eNM5`，venue `ICML 2026 regular`。字段只有 `pdf` 与 `supplementary_material`，没有 software/code 字段 |
| OpenReview PDF | https://openreview.net/pdf?id=qX4Nm7eNM5 | HTTP 403 |
| OpenReview 补充 zip | API 值 `/attachment/42ca8e0fc7d55b0037132aea60f28d47416d2336.zip`；`attachment?id=qX4Nm7eNM5&name=supplementary_material` | HTTP 403，`ChallengeRequiredError`。**zip 未下载、未列文件**。因此不能把它写成官方代码，也不能证明里面没有代码。附件状态：`INACCESSIBLE` |
| GitHub | 仓库检索 `2601.20386`、`SCORE-SAFFRON`、`donation` 无关项 | 三个查询 `total_count=0`。`Overshoot Refund` 的 code search 为 401 |
| 作者 GitHub | 用户检索 “Qi Kuang” 出现 `QiKuang1999` | 该用户公开仓是 Datathon 与个人主页，没有 SCORE 仓。不能据此认定他就是本文作者。Bowen Gang 用户检索 `total_count=0`。Yin Xia 的 OpenReview 个人页 https://openreview.net/profile?id=~Yin_Xia4 的 Personal Links 为空，未列出 SCORE 代码 |
| 错配仓 | https://github.com/score-spec/score-implementation-sample | 描述 “Score implementation template”，Apache-2.0，与 ICML SCORE 无关 |
| 错配仓 | https://github.com/OliverHennhoefer/online-fdr | 第三方在线 FDR 包，BSD-3-Clause，homepage https://oliverhennhoefer.github.io/online-fdr/ 。不是 Kuang/Gang/Xia 的官方实现 |
| Zenodo | `q=2601.20386` | `total=0` |

心梗数据（论文引用，不是代码）：https://figshare.le.ac.uk/articles/dataset/Myocardial_infarction_complications_Database/12045261

**结论：官方仓库 `NOT_FOUND`。** OpenReview 补充 zip 另标 `INACCESSIBLE`。实现模式 `INDEPENDENT_REIMPLEMENTATION`。

## 3. UAI 2026 — online e-closure / donation

全名：Improving online FDR procedures via online analogs of e-closure and compound e-values。Ziyu Xu, Lasse Fischer, Aaditya Ramdas。UAI 2026，PMLR v337。**不是 CCF-A Main。** arXiv:2603.24792。

正文声称有代码。arXiv 源码 `arxiv_flat.tex` 第 142 行写的是 “We also provide a user-friendly implementation of our methods in the accompanying code”，同一句没有 URL。HTML https://arxiv.org/html/2603.24792v3 同样只有这句话。

| 来源 | URL | 本轮看到的内容 |
| --- | --- | --- |
| PMLR | https://proceedings.mlr.press/v337/xu26a.html | 只有 PDF：https://raw.githubusercontent.com/mlresearch/v337/main/assets/xu26a/xu26a.pdf |
| PMLR 仓库目录 | GitHub git tree `mlresearch/v337` | 匹配 `xu26a` 的路径只有 `assets/xu26a` 与 `assets/xu26a/xu26a.pdf` |
| 猜测的补充 PDF | https://raw.githubusercontent.com/mlresearch/v337/main/assets/xu26a/xu26a-supp.pdf | HTTP 404 |
| arXiv 源码 | https://arxiv.org/e-print/2603.24792 （gzip，2026-07-09） | 解包为 `arxiv_flat.tex`、`figures/*.pdf`。有 “accompanying code” 句子，**没有** github / zenodo / SWH URL。图是 PDF，不是脚本 |
| OpenReview | https://openreview.net/forum?id=QiiA0VJpIG | API 笔记 id `QiiA0VJpIG`，venue `UAI 2026 Poster`。有 `pdf`、`supplementary_material`（`.pdf`）、`latex_source_zip`。没有 software/code 字段 |
| OpenReview 附件 | `supplementary_material` = `/attachment/f5b5062e7595529e7b9edaccc1d9ac21f34654cf.pdf`；latex zip = `/attachment/fb59727f0c6b12543800805eafd7db178de35039.zip` | 匿名下载 HTTP 403。附件状态 `INACCESSIBLE`。arXiv 源码已经覆盖正文和附录，且其中没有代码 URL |
| Xu 主页 | https://neilzxu.me/ | Publications 列出本文，只有 `[arXiv 2603.24792]`。同页其他论文有 `[code]`，本文没有 |
| Fischer 主页 | https://fischerl5.wordpress.com/ | Publications 列出本文（UAI）。Software 只指向 https://github.com/fischer23 ，没有单独的本文链接。邮箱 `fischer1@uni-bremen.de` |
| Ramdas 主页 | https://www.stat.cmu.edu/~aramdas/ 跳转到 https://web.stanford.edu/~aramdas/ | 条目为 “Improving online FDR procedures … UAI, 2026. arXiv”。相邻论文常写 `\| code`，这一条没有 code。主页邮箱写法为 `aramdas@stanford.edu`。论文脚注仍是 `aramdas@cmu.edu` |
| Ramdas CV | https://stat.cmu.edu/~aramdas/AadityaRamdasCV.pdf | 第 80 条列出本文，摘录中无代码 URL |
| GitHub 仓库检索 | `2603.24792`、`donation e-LOND`、`online e-closure` | 前两个 `total_count=0`。`online e-closure` 唯一命中是无关的 Swift 教程 `devACADEMY-it/Funzioni-e-Closure-in-Swift` |
| Zenodo | `q=2603.24792` | `total=0` |

### Ziyu Xu（`neilzxu`）全部 30 个公开仓

API：https://api.github.com/users/neilzxu/repos?per_page=100 （`public_repos=30`）。与本文算法名字相近、但**不是** UAI 2026 online donation / online e-closure 的仓：

| 仓库 | 实际对应 | 为何不是本文 |
| --- | --- | --- |
| https://github.com/neilzxu/eclosure | 离线 *Bringing Closure to False Discovery Rate Control*，arXiv:2509.02517 | homepage 指向该 arXiv。默认分支 `release`。最新 commit `8bbd5a37b48abb6d79f5e1eef95d4fe6315339b5`，2026-03-23，subject `version update`。已在本地浅克隆，见 §6。API license 字段为空；PyPI `eclosure` 0.1.1 写 GPL-3.0-only |
| https://github.com/neilzxu/evalue-omt | AISTATS 2024 *Online multiple testing with e-values*，arXiv:2311.06412 | 默认分支 `publishable`。commit `cda6e9962514924044143920cf1cadbec8738d52`，2024-03-08，`Initial code release`。已在本地浅克隆 |
| https://github.com/neilzxu/e_bmt | NeurIPS 2021 bandit multiple testing，arXiv:2107.07322 | 见 §1 |
| https://github.com/neilzxu/suplord | 2021 *Dynamic Algorithms for Online Multiple Testing*，arXiv:2010.13953 | 默认分支 `main`，最后 push 2021-08-20 |
| https://github.com/neilzxu/online_e-BH | *An online generalization of the e-BH procedure*，arXiv:2407.20683 | README 写明该文。MIT。最后 push 2024-07-30。同内容也在 https://github.com/fischer23/online_e-BH |
| https://github.com/neilzxu/closed-ebh | *Bringing closure to FDR control: beating the e-Benjamini-Hochberg procedure* | homepage https://arxiv.org/abs/2504.11759 。离线 e-BH，不是 UAI 在线 donation |
| https://github.com/neilzxu/rand-ebh | 随机化 e-BH，arXiv:2305.11126 | homepage 指向该文。Bernoulli 2026 那篇，主页单独给了 `[code]`，不是 UAI 2026 |

其余公开仓（`active-rcps`、`rcps`、课程作业、dotfiles、博客、`zotcite` 等）与 donation e-LOND / online e-closure 无关。

### Lasse Fischer（`fischer23`）7 个公开仓

https://github.com/fischer23 。没有名为 donation、e-closure 或 2603.24792 的仓。

| 仓库 | 实际对应 |
| --- | --- |
| https://github.com/fischer23/Closed-Online-Procedures | *The Online Closure Principle*，Annals of Statistics 2024。README 写明 ADDIS-Spending，并引用 Zenodo https://doi.org/10.5281/zenodo.2396572 （Robertson 等的 IMPC 数据，不是本文） |
| https://github.com/fischer23/online_e-BH | arXiv:2407.20683，不是 UAI 2026 |
| https://github.com/fischer23/online_true_discovery | README：*Admissible online closed testing must employ e-values*。不是 donation e-LOND |
| `Adaptive-Discard-Graph`、`Exhaustive-ADDIS-procedures`、`boosting_SPRT`、`MC-testing-by-betting` | 名称对应 Fischer 的其他论文，不是 arXiv:2603.24792 |

用户检索 “Lasse Fischer” 还看到 `Fischer96/rack`（Rust 音频插件）和空的 `lasficw`，与本文无关。

**结论：`INCOMPLETE`。** 论文写了 accompanying code，本轮在出版社页、arXiv 源码、作者主页和上述公开仓里都没有定位到该实现。实现模式仍是 `INDEPENDENT_REIMPLEMENTATION`，不能标 `OFFICIAL`。OpenReview 的 PDF/LaTeX 附件是 `INACCESSIBLE`，不改变这条代码结论：已读到的 arXiv 源码里没有仓库地址。

## 4. arXiv:2609.26651 — e-PS（预印本）

全名：Sample-Efficient Multiple Testing with Adaptive Data Collection。Zhanran Lin, Wanteng Ma, Zhimei Ren, Yuting Wei。宾夕法尼亚大学。**预印本，不是已发表 CCF-A。** 算法名 e-PS（e-value-based posterior sampling）。

| 来源 | URL | 本轮看到的内容 |
| --- | --- | --- |
| arXiv abs | https://arxiv.org/abs/2609.26651 | 无 code 栏。提交信息为 2026-09-22 |
| arXiv HTML | https://arxiv.org/html/2609.26651v1 | 无 GitHub。水印实验写的是 “we generate” OPT-1.3B 流 |
| arXiv 源码 | https://arxiv.org/e-print/2609.26651 （gzip，2026-09-23） | `main.tex` + `appendix.tex` + `images/*.png`。唯一 URL 是 Jester 脚注。没有作图脚本，没有代码发布句 |
| OpenReview | API 检索 “Sample-Efficient Multiple Testing with Adaptive Data Collection e-PS” | 返回的笔记没有这篇的 title/authors。未见 forum |
| 作者主页 | https://wantengma.github.io/ | 预印本第 2 条就是本文，无 code 链接。邮箱 `wanteng@wharton.upenn.edu` |
| 作者主页 | https://zhimeir.github.io/ | 未在抓取到的首页正文里单列本文代码。邮箱 `zren@wharton.upenn.edu`。其 20 个公开仓是 conformal / knockoff 等，没有 e-PS |
| 作者主页 | https://yutingwei.github.io/ | 邮箱 `ytwei@wharton.upenn.edu`。唯一公开仓是个人主页 |
| GitHub 用户 | `Zhanran Lin` 检索 `total_count=0` | 未找到主页，也没有公开邮箱 |
| GitHub 仓库 | `2609.26651`、`e-value posterior sampling` | `total_count=0`。`ePSU`、`ePSproc`、`e-PSHAD` 是无关项目 |
| Zenodo | `q=2609.26651` | `total=0` |

### Jester 评分

论文脚注：https://www.kaggle.com/datasets/vikashrajluhaniwal/jester-17m-jokes-ratings-dataset

2026-11-02 打开该数据集页，能看到 `jester_items.csv`（58.77 kB）和 `jester_ratings.csv`，Data Explorer 合计 34.55 MB，评分约 170 万、150 个笑话、范围 \([-10,10]\)。页上写来源是 http://eigentaste.berkeley.edu/dataset/ 。这是公开数据集卡片，本轮**没有下载** csv。Kaggle 实际取文件通常要登录；curl 对同一 URL 发 HEAD 曾得到 404，与浏览器式抓取不一致，以数据集页正文为准。

Berkeley 页面 https://eigentaste.berkeley.edu/dataset/ 本轮 WebFetch 504，直连 zip 时 TLS 失败，所以没有复核 `jester_dataset_2.zip` 是否仍可直接下载。检索索引里的归档页 https://eigentaste.berkeley.edu/dataset/archive/ 列有 Dataset 2（约 170 万条、150 个笑话）。论文实验用 \(K=140\) 个笑话，和 150 列原始表不是同一个文件形状。

### OPT-1.3B 水印流

§7.2 写：作者自己用 OPT-1.3B 生成五组、每组 1000 条流（200 条 Gumbel-max 水印 + 800 条无水印），20 个 prompt，温度在 \([0.5,0.8]\) 上 5 个等距点，长度 200–1200 token。源码和 HTML 都没有这些流的下载地址。

所引 Su, Wang, Zhao，arXiv:2602.14286 https://arxiv.org/abs/2602.14286 的 HTML 也没有 GitHub 或数据链接；他们的 OPT-1.3B 设置是温度 0.5 与 1.0，和 e-PS 的五温度网格不是同一份预生成数据。`jthickstun/watermark`、`THU-BPM/MarkLLM`、`xiaojunxu/learning-to-watermark-llm` 会提到 OPT-1.3B，但是别的论文的生成代码，不是这五组流。本轮没有下载 OPT-1.3B，也没有装 vLLM/CUDA。

**结论：代码 `NOT_FOUND`。** 实现模式 `INDEPENDENT_REIMPLEMENTATION`。Jester 评分公开可查。作者的 OPT-1.3B 预生成流未发布；若以后要复现 Figure 4，缺的是这批数据，不能用别的模型输出代替。

## 5. 明确排除的仓库

| 仓库 | 排除原因 |
| --- | --- |
| https://github.com/neilzxu/e_bmt | NeurIPS 2021，不是 AAAI 2026 |
| https://github.com/neilzxu/eclosure `8bbd5a37b48abb6d79f5e1eef95d4fe6315339b5` | 离线 arXiv:2509.02517，不是 UAI 2026 在线 e-closure |
| https://github.com/neilzxu/evalue-omt `cda6e9962514924044143920cf1cadbec8738d52` | AISTATS 2024 |
| https://github.com/neilzxu/suplord | 2021 dynamic online testing |
| https://github.com/score-spec/score-implementation-sample | 与 ICML SCORE 无关 |
| https://github.com/OliverHennhoefer/online-fdr | 第三方包，不是作者代码 |
| https://github.com/fischer23/Closed-Online-Procedures | Annals of Statistics 2024 的 online closure principle，不是 UAI 2026 donation |
| https://github.com/fischer23/online_e-BH 与 `neilzxu/online_e-BH` | arXiv:2407.20683 |
| https://github.com/fischer23/online_true_discovery | admissible online closed testing，不是本文 |

## 6. 已有 v0 审计与本地克隆

对照 `idea_C/reports/v0/BASELINE_AUDIT.md`（审计日期 2026-11-01）。v0 的四条 “无本文官方仓” 与本轮一致。本轮新增的是：作者全量公开仓、arXiv 源码解包、OpenReview 笔记、Jester 数据集页、水印流未发布，以及 OpenReview 附件 403。

本地克隆与 MANIFEST 仍一致，本轮没有改它们：

- `idea_C/baselines/eclosure/official/`：`git rev-parse HEAD` = `8bbd5a37b48abb6d79f5e1eef95d4fe6315339b5`，2026-03-23 17:05:48 -0400，`version update`。记录在 `idea_C/baselines/eclosure/OFFICIAL_MANIFEST.md`。`.gitignore` 已忽略该目录。
- `idea_C/baselines/eclosure/evalue-omt/`：`cda6e9962514924044143920cf1cadbec8738d52`，2024-03-08 23:55:23 +0000，`Initial code release`。记录在 `idea_C/baselines/eclosure/EVALUE_OMT_MANIFEST.md`。同样被 gitignore。

这两棵树都**不是** UAI 2026 在线方法的官方代码。本轮没有新克隆，所以没有新的 fetch 脚本。

## 7. 索要代码的邮件草稿（未发送）

下面只是草稿。没有发送。

### AAAI 2026

收件人：`liuweiwei863@gmail.com`（2026-11-02 从其主页读到）。`idea_C/reports/v0/BASELINE_AUDIT.md` 还写了 Zhengyu Zhou 的 `zzysince1999@gmail.com`；本轮 AAAI PDF 没有下载成功，这个地址没有从 PDF 再核对。若仍有效，可抄送。

```text
Subject: Code for AAAI 2026 "On the Robustness of Bandit Multiple Testing"

Dear Professor Liu and Zhengyu Zhou,

I am reproducing the experiments in your AAAI 2026 paper "On the Robustness of Bandit Multiple Testing" (https://ojs.aaai.org/index.php/AAAI/article/view/40148). I could not find a public repository linked from the AAAI page, ACM record, or your homepage.

If you released code or simulation scripts, could you share the repository, commit, and any note on the Lévy contamination parameters used in the figures? I will not redistribute the code against its license.

Thank you,
Grant Wright
```

### SCORE

收件人：`xiayin@fudan.edu.cn`。未找到 Kuang 与 Gang 的公开邮箱。

```text
Subject: Code for ICML 2026 SCORE (arXiv:2601.20386)

Dear Professor Xia, Qi Kuang, and Bowen Gang,

I am reproducing "SCORE: A Unified Framework for Overshoot Refund in Online FDR Control" (PMLR v306, https://proceedings.mlr.press/v306/kuang26c.html, arXiv:2601.20386). The PMLR page and the arXiv source do not contain a repository URL. The OpenReview supplementary zip (forum https://openreview.net/forum?id=qX4Nm7eNM5) was not anonymously downloadable when I tried.

If an official implementation exists, could you point me to the repository and commit, and say whether the OpenReview zip is the appendix, the code, or both?

Thank you,
Grant Wright
```

### UAI 2026

收件人：论文脚注 `xzy@cmu.edu`、`fischer1@uni-bremen.de`；Ramdas 现主页为 `aramdas@stanford.edu`（论文脚注仍写 `aramdas@cmu.edu`）。

```text
Subject: Accompanying code for UAI 2026 online e-closure (arXiv:2603.24792)

Dear Ziyu Xu, Lasse Fischer, and Professor Ramdas,

Your UAI 2026 paper "Improving online FDR procedures via online analogs of e-closure and compound e-values" (https://proceedings.mlr.press/v337/xu26a.html, arXiv:2603.24792) says there is a user-friendly implementation in the accompanying code. I could not find a URL in the arXiv source, on the PMLR page, or on your homepages.

The public repositories I did find are for other papers, and I am not treating them as this release: neilzxu/eclosure at 8bbd5a37 (offline arXiv:2509.02517), neilzxu/evalue-omt at cda6e996 (AISTATS 2024), neilzxu/suplord (2021), and fischer23/online_e-BH (arXiv:2407.20683).

Could you share the repository and commit for the online closed and donation procedures in this UAI paper?

Thank you,
Grant Wright
```

### e-PS

收件人：`wanteng@wharton.upenn.edu`，`zren@wharton.upenn.edu`，`ytwei@wharton.upenn.edu`。未找到 Zhanran Lin 的公开邮箱。

```text
Subject: Code and watermark streams for arXiv:2609.26651 (e-PS)

Dear Zhanran Lin, Wanteng Ma, Professor Ren, and Professor Wei,

I am reproducing "Sample-Efficient Multiple Testing with Adaptive Data Collection" (arXiv:2609.26651). I did not find a public code repository. The Jester ratings page cited in the paper is still up. I did not find the five pregenerated OPT-1.3B watermark pools used in Figure 4 (20 prompts, five temperatures in [0.5, 0.8], lengths 200–1200).

If you can share the repository, and separately the watermark streams or a checksum for them, I would not need to regenerate text with OPT-1.3B.

Thank you,
Grant Wright
```

## 8. 本轮没有做的事

- 没有浅克隆新仓库。
- 没有修改上游源码，没有安装 CUDA / vLLM。
- 没有把 UAI 写成 CCF-A，没有把 e-PS 写成已发表 CCF-A。
- 没有发邮件。
