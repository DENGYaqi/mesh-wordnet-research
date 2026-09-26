# 定义语义评分：第一轮独立实验

## 边界

输入是已保存运行中的 3,018 对候选，不搜索新配对，也不改变历史 SIF、167 条邻近结果或展示页。输出只是重新排序的**待核查关系**。概念融合需要人工确认等价，并在后续步骤检查层级及实际写入；本程序不做这些事。

## 输入与模型

- 默认候选：`research/runs/20260926T052631818279Z/candidate_scores.csv`。每条保留原始数据行号，以及 MeSH UI、WordNet synset、双方名称和原始英文定义。
- 抽样用词典：`research/data/data_complet_WN(v3).csv`。将同名且该 WordNet 名称在完整名词表中有多个 synset 的配对划入“可能多义”，这只是抽样线索，不是错配标签。
- 语义模型：`sentence-transformers/all-MiniLM-L6-v2` 的 [Qdrant ONNX 转换](https://huggingface.co/qdrant/all-MiniLM-L6-v2-onnx)，固定仓库提交 `5f1b8cd78bc4fb444dd171e59b18f3a3af89a079`。FastEmbed 0.8.1 在 CPU 上以同一模型编码两侧定义，向量归一化后计算余弦相似度。固定转换的 tokenizer 实际最多接收 128 个 token；超出的原始行号写入 manifest 和 CSV，不把截断结果冒充完整定义比较。模型文件 SHA-256 也写入 manifest。
- 基线：两侧定义放入同一个 TF-IDF 词表，小写、单词及二元词组，行向量归一化后取点积。

`research/semantic-requirements.txt` 列出直接依赖，`research/semantic-requirements.lock` 固定容器安装的全部依赖。模型权重与所有运行结果均被 Git 忽略。下载失败、输入缺失或字段不符时，运行目录的 `manifest.json` 会保留失败状态与原因。

## 运行与人工审核

从项目根目录执行：

```powershell
docker compose --profile semantic build semantic
docker compose --profile semantic run --rm semantic
```

首次运行需要联网取模型；模型已在 `research/models/semantic-minilm-l6-v2/` 后，评分可离线重跑。每次运行新建 `research/runs/semantic-.../`：

- `review_sample.csv`：评分前固定种子 42 抽 120 对，三组各 40；*Air* 作为预先指定的同名多义例子包含在第三组。它不含模型分数，避免按分数挑审核样本。
- `ranked_candidates.csv`：3,018 对的语义排序，含 TF-IDF 对照、原行号及待核查状态。分数可以为负值，并非概率。
- `manifest.json`：输入和模型校验值、固定版本、依赖、两侧截断行、运行状态及分数范围。

只编辑审核表的 `review_relation`、`review_note` 两列。关系必须从“等价 / MeSH 更宽 / WordNet 更宽 / 相关但不等价 / 无关 / 不确定”中选；“不确定”不计入二分类排序指标。填好全部 120 条后运行：

```powershell
docker compose --profile semantic run --rm semantic python score_semantic.py evaluate --run-dir /research/runs/semantic-实际目录名
```

`review_evaluation.json` 分整体及三组报告 average precision，列出高分非等价和低分等价例子。报告仅说明这个审核样本上的排序表现，不能直接推广为全量准确率。只有模型优于 TF-IDF 且错误案例可接受，才考虑后续页面展示；否则保留实验记录，不称为改进。未经人工标注时，程序拒绝生成评估。

## 已知限制

英文通用模型未针对 MeSH 或生物医学词义微调；高语义相似不必然是同一概念。TF-IDF 也只能作为基础对照。词典中的同名多义数量是抽样代理指标，不能代替人工标注。历史邻近 167 条只是辅助证据，没有被作为标准答案。

## 本机核验记录（2026-09-26）

- 固定输入的 3,018 条候选标识与双方原定义按 `source_row` 逐行核对一致；抽样池分别为 2,069 / 605 / 344 条，各抽 40 条。
- 语义相似度范围约为 -0.0665 到 0.9561，全部为有限值；`Air` 的同名异义配对约为 0.0934，位列 2,929。这个例子仍需人工填写关系标签。
- 固定模型的 128 token 上限导致 71 条 MeSH 定义截断，WordNet 定义 0 条；这些行号与逐行截断标记都在运行产物中。
- 两次完整运行的排名文件及审核抽样文件 SHA-256 分别完全相同。最终运行目录为 `research/runs/semantic-20260926T112518831085Z/`，排名文件 SHA-256 为 `74ec1fef7e091af0e7e0dc0a26109a18bca2ec8780215b8a7baa9800ab1be8b2`。
- 未填写审核关系时，评估命令拒绝报告指标。缺少输入时，运行清单写入失败状态和原因。尚无人工审核结果，因此**没有排序效果或“改进”结论**。
