# MeSH–WordNet 概念匹配实验

这个项目复现 2019 年 Yseop 实习中的概念匹配实验。历史 Notebook 的代码保存在 `research/notebooks/`；公开版本移除了运行输出，带输出的原件仅保存在本机。可执行入口是 `research/run_historical.py`。`viewer/` 只读取已保存的实验结果，展示候选配对和邻近概念关系。

这个项目独立运行。实验结果没有写入 Lexicon 词库数据库；概念融合及质量评估当年没有完成。

## 打开展示页

先启动 Docker Desktop，在本目录运行：

```powershell
docker compose up -d --build viewer
```

打开 [本机展示页](http://127.0.0.1:18082/)。页面固定读取私有的 Python 3.12.0 运行目录 `research/runs/20260926T052631818279Z/` 中的 `manifest.json`、`candidate_scores.csv` 和 `neighbor_matches.csv`；Python 3.7.17 的旧运行目录仍保留。若复制项目时未带上这个被 Git 忽略的目录，页面会提示缺少结果；运行下面的 `all` 生成新目录后，将 `compose.yaml` 中的 `RESEARCH_RUN_DIR` 改为该目录名，再重启 viewer。停止页面用 `docker compose down`。

展示页仍使用独立的 Python 3.11 容器；可执行的实验流程已升级为 Python **3.12.0**。本机无需预装这两套环境。服务只监听 `127.0.0.1:18082`。概览页显示的 Python 版本来自所展示运行目录的 `manifest.json`，不是展示容器自身的版本；运行目录的原始编号可在 `compose.yaml` 中查看。

在“候选配对”中可搜索并点击一行查看两侧定义，也可用“全选当前结果”选择当前搜索结果，下载所选记录的完整 CSV。更改搜索词后，选择范围会随之重置。

在“邻近概念 → 关系图”中，可按 MeSH 名称或 ID、WordNet synset 找到一条结果。默认图显示它及共享节点的邻近记录；打开“查看全部 167 条关系”可浏览全图。拖动节点、缩放和平移图面，点击节点可看归档定义和关联记录；“表格”视图保留原始五列结果。绿色／蓝色实线箭头分别是 MeSH／WordNet 父子关系，紫色虚线是历史候选对应，橙色点线是父概念模糊匹配。跨词库连线均未经人工确认，不表示概念已经融合。图谱只读取已有 CSV，不需要 Neo4j 或联网服务。

## 核验与重训

```powershell
docker compose --profile training run --rm research python run_historical.py verify
docker compose --profile training run --rm research python -u run_historical.py all
```

`verify` 从 MeSH XML 与 WordNet 3.0 词典重新提取全量表，核对归档的 **3,018** 条候选和 **167** 条邻近概念结果。`all` 还会按历史规则预处理、分别重训两套 300 维 Word2Vec，并生成新运行目录；归档的 `research/data/`、`research/models/` 和原运行目录不会被覆盖。首次构建需联网获取固定版本依赖与 NLTK `punkt_tab`；运行容器以 `research/requirements.lock` 为准，`research/requirements.txt` 列出直接依赖。重训可能耗时较长。

候选阶段使用归档的 WordNet V1 筛选集，因为现存 Notebook 无法从原始词典完整重建它。旧训练的随机状态和当年 spaCy 模型版本未留存；新版 spaCy 与 Gensim 的重训向量、训练句数和词表不保证与旧版相同。Python 3.12.0 运行已核对候选标识和定义、邻近结果逐行一致，数值差异见 [VERIFICATION.md](VERIFICATION.md)。历史 SIF 分数全部约为 1，**不能作为匹配准确率或排序依据**。报告中的 3,019 包含被误读的 CSV 表头；2,468 是 2,301 与 167 之和，并非已写入的融合概念数。

GitHub 仓库只包含代码和说明。原始数据、模型和运行结果均被 `.gitignore` 排除，保持本机私有；因此从公开仓库克隆后，展示页不会立即显示历史结果，复现命令也需要先备齐对应数据。详细核对记录见 [VERIFICATION.md](VERIFICATION.md)。

`research/notebooks/` 和 `research/requirements-notebooks.txt` 保留 2019 年的查阅档案，不属于本次 Python 3.12 可执行流程。
