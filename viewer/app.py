"""Local, read-only presentation of the 2019 MeSH–WordNet experiment."""

import json
import os
from pathlib import Path

import streamlit as st

from results import ResultsError, load_results, search_candidates


RUN_DIR = Path(os.environ.get("RESEARCH_RUN_DIR", "/runs/20260924T161334509381Z"))

st.set_page_config(page_title="MeSH × WordNet | Yseop 2019", page_icon="🔬", layout="wide")
st.title("MeSH × WordNet")
st.caption("2019 年 Yseop 实习实验 · 本机私有展示")


@st.cache_data(show_spinner=False)
def read_run(path: str):
    return load_results(Path(path))


try:
    manifest, candidates, neighbors = read_run(str(RUN_DIR))
except ResultsError as exc:
    st.error(str(exc))
    st.info("请先按项目 README 运行历史实验，并检查展示页配置的运行目录。")
    st.stop()

overview, candidate_tab, neighbor_tab, method_tab = st.tabs(
    ["项目概览", "候选配对", "邻近概念", "方法与局限"]
)

with overview:
    st.subheader("从两个词库寻找可能对应的概念")
    st.write(
        "实验提取 MeSH 与 WordNet 概念，分别训练词向量，按历史规则生成候选配对，"
        "再比较部分父子概念关系。本页展示保存下来的实验结果，供逐条查阅。"
    )
    left, middle, right = st.columns(3)
    left.metric("候选配对", f"{len(candidates):,}")
    middle.metric("邻近概念结果", f"{len(neighbors):,}")
    right.metric("历史分数接近 1", f"{manifest.get('scores_near_one', 0):,}")
    st.warning(
        "历史 SIF 分数几乎都等于 1，不能作为匹配准确率，也不能用于候选排序。"
        "这些是待核查的候选，不是已完成的概念融合。"
    )
    st.markdown("**阅读顺序**　候选配对 → 两侧定义 → 邻近概念关系 → 方法与局限")
    st.caption(f"展示运行：{RUN_DIR.name} · Python {manifest.get('python', '未知')}")

with candidate_tab:
    st.subheader("浏览候选配对")
    query = st.text_input(
        "搜索 MeSH 名称或 ID、WordNet 名称或 synset",
        placeholder="例如 Abattoirs、M0000003 或 abattoir.n.01",
    )
    filtered = search_candidates(candidates, query).reset_index(drop=True)
    st.caption(f"找到 {len(filtered):,} / {len(candidates):,} 条；点击一行查看两侧定义。")
    if filtered.empty:
        st.info("没有找到符合条件的候选。请换一个名称、ID 或 synset 试试。")
    else:
        display = filtered[["MeSH_UI", "Name_MeSH", "Name_WN", "WN_synset"]].rename(
            columns={
                "MeSH_UI": "MeSH ID",
                "Name_MeSH": "MeSH 名称",
                "Name_WN": "WordNet 名称",
                "WN_synset": "WordNet synset",
            }
        )
        event = st.dataframe(
            display,
            hide_index=True,
            width="stretch",
            height=380,
            on_select="rerun",
            selection_mode="single-row",
            key="candidate_table",
        )
        selected = event.selection.rows
        if selected and selected[0] < len(filtered):
            row = filtered.iloc[selected[0]]
            st.divider()
            st.subheader(f"{row['Name_MeSH']} ↔ {row['Name_WN']}")
            mesh, wordnet = st.columns(2)
            with mesh:
                st.markdown("**MeSH**")
                st.caption(f"ID：{row['MeSH_UI']}")
                st.write(row["def_MeSH"].strip() or "原记录无定义")
            with wordnet:
                st.markdown("**WordNet**")
                st.caption(f"Synset：{row['WN_synset']}")
                st.write(row["def_WN"].strip() or "原记录无定义")
            st.caption(f"历史 SIF 原始值：{row['Similarity_highest']} · 此值不能用于排序或判断正确性")
        else:
            st.info("选择表格中的一行，查看概念详情。")

with neighbor_tab:
    st.subheader("邻近概念关系")
    st.write("这里记录了历史流程中识别出的 167 条父子概念比较结果。")
    event = st.dataframe(
        neighbors[
            [
                "identify_father_MeSH",
                "identify_child_MesH",
                "synset_father_wn",
                "synset_child_wn",
                "Parti_Ratio(fuzz)_between_father_concept",
            ]
        ].rename(
            columns={
                "identify_father_MeSH": "MeSH 父概念",
                "identify_child_MesH": "MeSH 子概念",
                "synset_father_wn": "WordNet 父概念",
                "synset_child_wn": "WordNet 子概念",
                "Parti_Ratio(fuzz)_between_father_concept": "历史模糊匹配值",
            }
        ),
        hide_index=True,
        width="stretch",
        height=380,
        on_select="rerun",
        selection_mode="single-row",
        key="neighbor_table",
    )
    selected = event.selection.rows
    if selected and selected[0] < len(neighbors):
        row = neighbors.iloc[selected[0]]
        wn_parent = json.dumps(row["synset_father_wn"], ensure_ascii=False)
        wn_child = json.dumps(row["synset_child_wn"], ensure_ascii=False)
        mesh_parent = json.dumps(row["identify_father_MeSH"], ensure_ascii=False)
        mesh_child = json.dumps(row["identify_child_MesH"], ensure_ascii=False)
        graph = f"""digraph G {{
          rankdir=LR; graph [bgcolor=transparent, pad=0.25];
          node [shape=box, style="rounded,filled", color="#d3dae4", fillcolor="#f5f7fa"];
          subgraph cluster_wordnet {{ label="WordNet"; color="#6e90bf";
            wn_parent [label={wn_parent}]; wn_child [label={wn_child}]; wn_parent -> wn_child; }}
          subgraph cluster_mesh {{ label="MeSH"; color="#7aa995";
            mesh_parent [label={mesh_parent}]; mesh_child [label={mesh_child}]; mesh_parent -> mesh_child; }}
        }}"""
        st.graphviz_chart(graph, width=760)
        st.caption(
            "历史父概念名称模糊匹配值："
            f"{row['Parti_Ratio(fuzz)_between_father_concept']}。"
            "该关系图表示原记录中的局部层级，不表示概念已融合或人工确认。"
        )
    else:
        st.info("选择一行，查看两侧的父子关系。")

with method_tab:
    st.subheader("方法、来源与结果边界")
    st.markdown(
        "1. 从 MeSH XML 和 WordNet 3.0 提取概念；候选阶段使用归档的 WordNet 筛选集。\n"
        "2. 按 2019 年 Notebook 规则预处理，并分别重训两套 300 维 Word2Vec。\n"
        "3. 计算历史 SIF 分数、生成候选，再比较邻近概念。"
    )
    st.write(
        "归档 WordNet 筛选集无法由现存 Notebook 完整重建。旧训练的随机种子与 spaCy 模型版本"
        "未留存，因此重训模型不保证与当年逐位相同。两套向量空间独立训练，历史 SIF 又逐对去主成分"
        "并取绝对余弦，造成分数退化。"
    )
    st.write(
        "报告中的 3,019 包含误读的 CSV 表头，实际候选为 3,018 条。后续另有一条被表头错误替代，"
        "因此进入邻近概念步骤的配对为 3,017 条；当年报告的 2,468 是 2,301 与 167 之和，"
        "不是已写入的融合概念数。融合写入及质量评估尚未完成。"
    )
    st.caption("页面只读取本机运行 CSV 与 manifest，不执行训练或新的匹配计算。")
