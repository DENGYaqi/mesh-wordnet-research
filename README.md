# MeSH–WordNet 概念匹配实验

这个项目复现 2019 年 Yseop 实习中的概念匹配实验。历史 Notebook 的代码保存在 `research/notebooks/`；公开版本移除了运行输出，带输出的原件仅保存在本机。可执行入口是 `research/run_historical.py`。`viewer/` 只读取已保存的实验结果，展示候选配对和邻近概念关系。

这个项目独立运行。实验结果没有写入 Lexicon 词库数据库；概念融合及质量评估当年没有完成。

## 打开展示页

先启动 Docker Desktop，在本目录运行：

```powershell
docker compose up -d --build viewer
```

打开 [本机展示页](http://127.0.0.1:18082/)。页面固定读取私有运行目录 `research/runs/20260924T161334509381Z/` 中的 `manifest.json`、`candidate_scores.csv` 和 `neighbor_matches.csv`。若复制项目时未带上这个被 Git 忽略的目录，页面会提示缺少结果；运行下面的 `all` 生成新目录后，将 `compose.yaml` 中的 `RESEARCH_RUN_DIR` 改为该目录名，再重启 viewer。停止页面用 `docker compose down`。

展示页使用独立的 Python 3.11 容器，历史实验使用 Python 3.7 容器；本机无需预装这两套环境。服务只监听 `127.0.0.1:18082`。

## 核验与重训

```powershell
docker compose --profile training run --rm research python run_historical.py verify
docker compose --profile training run --rm research python -u run_historical.py all
```

`verify` 从 MeSH XML 与 WordNet 3.0 词典重新提取全量表，核对归档的 **3,018** 条候选和 **167** 条邻近概念结果。`all` 还会按历史规则预处理、分别重训两套 300 维 Word2Vec，并生成新运行目录；归档的 `research/data/` 和 `research/models/` 不会被覆盖。首次构建需联网获取固定版本依赖，重训可能耗时较长。

候选阶段使用归档的 WordNet V1 筛选集，因为现存 Notebook 无法从原始词典完整重建它。旧训练的随机状态和 spaCy 模型版本未留存，重训向量不保证逐位相同。历史 SIF 分数全部约为 1，**不能作为匹配准确率或排序依据**。报告中的 3,019 包含被误读的 CSV 表头；2,468 是 2,301 与 167 之和，并非已写入的融合概念数。

GitHub 仓库只包含代码和说明。原始数据、模型和运行结果均被 `.gitignore` 排除，保持本机私有；因此从公开仓库克隆后，展示页不会立即显示历史结果，复现命令也需要先备齐对应数据。详细核对记录见 [VERIFICATION.md](VERIFICATION.md)。
