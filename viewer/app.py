"""Local, read-only presentation of the 2019 MeSH–WordNet experiment."""

import os
from pathlib import Path

import ctranslate2
import pandas as pd
import sentencepiece as spm
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder
from streamlit_echarts import st_echarts

from results import (
    ResultsError, build_neighbor_graph, load_results, local_neighbor_rows,
    search_candidates, search_neighbors,
)


RUN_DIR = Path(os.environ.get("RESEARCH_RUN_DIR", "/runs/20260926T052631818279Z"))
REVIEWED_TRANSLATIONS = {
    "M0014446": (
        "指作为个体的人（如堕胎申请者），或作为某一群体成员的人（如西班牙裔美国人）。"
        "不用于描述各类专业人员（如医师）或职业人员（如图书馆员）；"
        "这类人员可使用“职业群体（Occupational Groups）”概念。"
    ),
    "M000630288": (
        "一种人格特质：面对威胁、挫折或失去时，倾向于产生愤怒、焦虑、沮丧、尴尬、悲伤等负面情绪。"
    ),
    "M000597380": "不吃肉的人。",
}

st.set_page_config(page_title="MeSH × WordNet | Yseop 2019", page_icon="🔬", layout="wide")
st.title("MeSH × WordNet")
st.caption("2019 年 Yseop 实习实验")


@st.cache_data(show_spinner=False)
def read_run(path: str):
    return load_results(Path(path))


@st.cache_resource(show_spinner=False)
def chinese_translator():
    model = Path(os.environ["TRANSLATION_MODEL_DIR"])
    return ctranslate2.Translator(str(model / "model"), device="cpu"), spm.SentencePieceProcessor(
        model_file=str(model / "sentencepiece.model")
    )


@st.cache_data(show_spinner=False)
def translate_definition(definition: str) -> str:
    translator, tokenizer = chinese_translator()
    tokens = tokenizer.encode(definition, out_type=str)
    result = translator.translate_batch([tokens], replace_unknowns=True, beam_size=4)[0]
    return "".join(result.hypotheses[0]).replace("▁", " ").strip()


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

with candidate_tab:
    st.subheader("浏览候选配对")
    query = st.text_input(
        "搜索 MeSH 名称或 ID、WordNet 名称或 synset",
        placeholder="例如 Abattoirs、M0000003 或 abattoir.n.01",
    )
    filtered = search_candidates(candidates, query).reset_index(drop=True)
    st.caption(f"找到 {len(filtered):,} / {len(candidates):,} 条；选择一行查看两侧定义，点击表头左上角可全选。")
    if filtered.empty:
        st.info("没有找到符合条件的候选。请换一个名称、ID 或 synset 试试。")
    else:
        table_key = f"candidate_table_{query}"
        display = filtered[["MeSH_UI", "Name_MeSH", "Name_WN", "WN_synset"]].rename(
            columns={
                "MeSH_UI": "MeSH ID",
                "Name_MeSH": "MeSH 名称",
                "Name_WN": "WordNet 名称",
                "WN_synset": "WordNet synset",
            }
        ).copy()
        display.insert(0, "选择", "")
        display["_row_index"] = range(len(display))
        grid = GridOptionsBuilder.from_dataframe(display)
        grid.configure_default_column(sortable=True, resizable=True)
        grid.configure_selection("multiple", use_checkbox=True, header_checkbox=True)
        grid.configure_column("选择", headerName="", width=48, pinned="left", sortable=False, resizable=False)
        grid.configure_column("MeSH ID", width=115)
        grid.configure_column("MeSH 名称", width=155)
        grid.configure_column("WordNet 名称", width=155)
        grid.configure_column("WordNet synset", flex=1, minWidth=210)
        grid.configure_column("_row_index", hide=True)
        grid_options = grid.build()
        grid_options["suppressRowClickSelection"] = False
        event = AgGrid(
            display,
            gridOptions=grid_options,
            height=380,
            theme="streamlit",
            update_on=["selectionChanged"],
            show_search=False,
            show_download_button=False,
            key=table_key,
        )
        selected_rows = event.selected_rows
        selected = selected_rows["_row_index"].astype(int).tolist() if isinstance(selected_rows, pd.DataFrame) else []
        if selected:
            st.caption(f"已选择 {len(selected):,} 条")
            st.download_button(
                "下载所选记录 CSV",
                filtered.iloc[selected].to_csv(index=False).encode("utf-8-sig"),
                file_name="mesh_wordnet_selected_candidates.csv",
                mime="text/csv",
            )
        if len(selected) == 1:
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
        elif not selected:
            st.info("选择表格中的一行，查看概念详情。")
        else:
            st.info("要查看两侧定义，请只选择一条记录。")

with neighbor_tab:
    st.subheader("邻近概念关系")
    st.write("这里记录了历史流程中识别出的 167 条父子概念比较结果。")
    graph_tab, table_tab = st.tabs(["关系图", "表格"])
    with graph_tab:
        query = st.text_input("搜索 MeSH 名称或 ID、WordNet synset", key="neighbor_search")
        matches = search_neighbors(neighbors, candidates, query)
        if matches.empty:
            st.info("没有找到符合条件的邻近概念记录，请换一个名称、ID 或 synset。")
        else:
            selected_index = st.selectbox(
                "选择一条邻近概念记录",
                matches.index.tolist(),
                format_func=lambda i: (
                    f"{neighbors.at[i, 'identify_father_MeSH']} → {neighbors.at[i, 'identify_child_MesH']}"
                    f"　｜　{neighbors.at[i, 'synset_father_wn']} → {neighbors.at[i, 'synset_child_wn']}"
                ),
            )
            full_graph = st.toggle("查看全部 167 条关系", key="full_neighbor_graph")
            visible = neighbors if full_graph else local_neighbor_rows(neighbors, selected_index)
            try:
                nodes, links = build_neighbor_graph(candidates, visible, show_labels=not full_graph)
            except ResultsError as exc:
                st.error(str(exc))
                st.stop()
            focal = neighbors.loc[selected_index]
            focal_nodes = {
                ("MeSH", focal["identify_father_MeSH"]), ("MeSH", focal["identify_child_MesH"]),
                ("WordNet", focal["synset_father_wn"]), ("WordNet", focal["synset_child_wn"]),
            }
            by_id = {node["id"]: node for node in nodes}
            for node in nodes:
                node["label"] = {"show": not full_graph and (node["source"], node["name"]) in focal_nodes}
            for link in links:
                if link["kind"] == "fuzzy":
                    link["label"]["show"] = (
                        not full_graph
                        and by_id[link["source"]]["name"] == focal["identify_father_MeSH"]
                        and by_id[link["target"]]["name"] == focal["synset_father_wn"]
                    )
            options = {
                "tooltip": {"trigger": "item"},
                "legend": {"data": ["MeSH", "WordNet"], "top": 0},
                "animationDuration": 300,
                "series": [{
                    "type": "graph", "layout": "force", "roam": True, "draggable": True,
                    "zoom": 0.85 if not full_graph else 0.65,
                    "data": nodes, "links": links,
                    "categories": [
                        {"name": "MeSH", "itemStyle": {"color": "#579775"}},
                        {"name": "WordNet", "itemStyle": {"color": "#628fc4"}},
                    ],
                    "label": {"show": False, "position": "right", "fontSize": 11},
                    "emphasis": {"focus": "adjacency"},
                    "force": {
                        "repulsion": 210 if not full_graph else 95,
                        "edgeLength": 135 if not full_graph else 75,
                        "layoutAnimation": not full_graph,
                    },
                }],
            }
            st.caption(
                "绿色为 MeSH，蓝色为 WordNet；实线箭头表示父子关系，紫色虚线表示历史候选对应，"
                "橙色点线表示父概念模糊匹配。跨词库连线均待核查，不表示已融合或人工确认。"
            )
            st.caption(f"当前图：{len(visible)} 条记录、{len(nodes)} 个节点。可拖动节点、滚轮缩放、拖动空白处平移；悬停或点击节点看详情。")
            chart_col, detail_col = st.columns([3, 1])
            with chart_col:
                event = st_echarts(
                    options=options, height="680px", theme="streamlit",
                    events={"click": "function(params) { return params.dataType === 'node' ? params.data.id : undefined; }"},
                    key="neighbor_graph",
                )
            selected_id = getattr(event, "chart_event", None) if event is not None else None
            default_id = "mesh:" + next(
                node["identifier"] for node in nodes
                if node["source"] == "MeSH" and node["name"] == neighbors.at[selected_index, "identify_child_MesH"]
            )
            node = by_id.get(selected_id, by_id[default_id])
            with detail_col:
                st.markdown(f"**{node['name']}**")
                st.caption(f"{node['source']} · {node['identifier'] or '归档候选表未收录 MeSH ID'}")
                if node["definition"]:
                    st.write(node["definition"])
                    translation = REVIEWED_TRANSLATIONS.get(node["identifier"])
                    if translation is None:
                        translation = translate_definition(node["definition"])
                    with st.container(border=True):
                        st.markdown("**中文翻译**")
                        st.write(translation)
                        if node["identifier"] not in REVIEWED_TRANSLATIONS:
                            st.caption("机器翻译，仅供参考；请以英文原文为准。")
                else:
                    st.info("归档候选表没有这个节点的定义。")
                st.markdown(f"**关联记录（{len(node['rows'])}）**")
                for index in node["rows"]:
                    row = neighbors.loc[index]
                    st.caption(
                        f"{row['identify_father_MeSH']} → {row['identify_child_MesH']} · "
                        f"{row['synset_father_wn']} → {row['synset_child_wn']} · "
                        f"模糊匹配 {row['Parti_Ratio(fuzz)_between_father_concept']}"
                    )
    with table_tab:
        st.dataframe(
            neighbors[
                [
                    "identify_father_MeSH", "identify_child_MesH",
                    "synset_father_wn", "synset_child_wn",
                    "Parti_Ratio(fuzz)_between_father_concept",
                ]
            ].rename(columns={
                "identify_father_MeSH": "MeSH 父概念",
                "identify_child_MesH": "MeSH 子概念",
                "synset_father_wn": "WordNet 父概念",
                "synset_child_wn": "WordNet 子概念",
                "Parti_Ratio(fuzz)_between_father_concept": "历史模糊匹配值",
            }),
            hide_index=True, width="stretch", height=450,
        )

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
