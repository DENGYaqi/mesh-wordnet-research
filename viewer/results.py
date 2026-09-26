"""Read one completed, immutable historical experiment run."""

import json
from pathlib import Path

import pandas as pd


CANDIDATE_COLUMNS = (
    "MeSH_UI",
    "Name_MeSH",
    "Name_WN",
    "WN_synset",
    "def_MeSH",
    "def_WN",
    "Similarity_highest",
)
NEIGHBOR_COLUMNS = (
    "synset_father_wn",
    "Parti_Ratio(fuzz)_between_father_concept",
    "synset_child_wn",
    "identify_father_MeSH",
    "identify_child_MesH",
)


class ResultsError(ValueError):
    """The selected run cannot be displayed faithfully."""


def load_results(run_dir: Path):
    run_dir = Path(run_dir)
    paths = {
        "manifest": run_dir / "manifest.json",
        "candidates": run_dir / "candidate_scores.csv",
        "neighbors": run_dir / "neighbor_matches.csv",
    }
    missing = [path.name for path in paths.values() if not path.is_file()]
    if missing:
        raise ResultsError(f"运行结果不完整，缺少：{', '.join(missing)}")

    try:
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        candidates = pd.read_csv(paths["candidates"], dtype=str, keep_default_na=False)
        neighbors = pd.read_csv(paths["neighbors"], dtype=str, keep_default_na=False)
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as exc:
        raise ResultsError(f"无法读取运行结果：{exc}") from exc

    if not isinstance(manifest, dict):
        raise ResultsError("manifest.json 格式错误：应为运行信息对象")

    for name, frame, expected in (
        ("candidate_scores.csv", candidates, CANDIDATE_COLUMNS),
        ("neighbor_matches.csv", neighbors, NEIGHBOR_COLUMNS),
    ):
        absent = [column for column in expected if column not in frame.columns]
        if absent:
            raise ResultsError(f"{name} 缺少字段：{', '.join(absent)}")

    if manifest.get("pairs") != len(candidates) or manifest.get("neighbors") != len(neighbors):
        raise ResultsError("CSV 行数与 manifest.json 的运行记录不一致")

    return manifest, candidates, neighbors


def search_candidates(candidates: pd.DataFrame, query: str) -> pd.DataFrame:
    query = query.strip()
    if not query:
        return candidates
    fields = ("MeSH_UI", "Name_MeSH", "Name_WN", "WN_synset")
    mask = pd.Series(False, index=candidates.index)
    for field in fields:
        mask |= candidates[field].str.contains(query, case=False, regex=False, na=False)
    return candidates.loc[mask]


def search_neighbors(neighbors: pd.DataFrame, candidates: pd.DataFrame, query: str) -> pd.DataFrame:
    query = query.strip().casefold()
    if not query:
        return neighbors
    mesh_ids = dict(zip(candidates["Name_MeSH"], candidates["MeSH_UI"]))
    fields = ("identify_father_MeSH", "identify_child_MesH", "synset_father_wn", "synset_child_wn")
    return neighbors.loc[
        [
            any(query in value.casefold() for value in (
                *(row[field] for field in fields),
                mesh_ids.get(row["identify_father_MeSH"], ""),
                mesh_ids.get(row["identify_child_MesH"], ""),
            ))
            for _, row in neighbors.iterrows()
        ]
    ]


def local_neighbor_rows(neighbors: pd.DataFrame, selected_index: int) -> pd.DataFrame:
    fields = ("identify_father_MeSH", "identify_child_MesH", "synset_father_wn", "synset_child_wn")
    selected = neighbors.loc[selected_index]
    related = {(field.startswith("synset_"), selected[field]) for field in fields}
    return neighbors.loc[
        [
            index for index, row in neighbors.iterrows()
            if any((field.startswith("synset_"), row[field]) in related for field in fields)
        ]
    ]


def build_neighbor_graph(candidates: pd.DataFrame, neighbors: pd.DataFrame, show_labels: bool):
    candidate_rows = candidates.to_dict("records")
    mesh = {row["Name_MeSH"]: row for row in candidate_rows}
    wn = {row["WN_synset"].removeprefix("Synset('").removesuffix("')"): row["def_WN"]
          for row in candidate_rows}
    candidate_pairs = {
        (row["Name_MeSH"], row["WN_synset"].removeprefix("Synset('").removesuffix("')"))
        for row in candidate_rows
    }
    nodes, links = {}, {}

    for index, row in neighbors.iterrows():
        mesh_parent, mesh_child = row["identify_father_MeSH"], row["identify_child_MesH"]
        wn_parent, wn_child = row["synset_father_wn"], row["synset_child_wn"]
        if mesh_child not in mesh or (mesh_child, wn_child) not in candidate_pairs:
            raise ResultsError(f"邻近结果第 {index + 1} 行无法与候选记录对应")
        try:
            ratio = int(row["Parti_Ratio(fuzz)_between_father_concept"])
        except ValueError as exc:
            raise ResultsError(f"邻近结果第 {index + 1} 行的模糊匹配值无效") from exc

        node_specs = (
            ("mesh:" + mesh[mesh_parent]["MeSH_UI"] if mesh_parent in mesh else "mesh:name:" + mesh_parent,
             mesh_parent, "MeSH", mesh[mesh_parent]["MeSH_UI"] if mesh_parent in mesh else "",
             mesh[mesh_parent]["def_MeSH"] if mesh_parent in mesh else ""),
            ("mesh:" + mesh[mesh_child]["MeSH_UI"], mesh_child, "MeSH", mesh[mesh_child]["MeSH_UI"], mesh[mesh_child]["def_MeSH"]),
            ("wn:" + wn_parent, wn_parent, "WordNet", wn_parent, wn.get(wn_parent, "")),
            ("wn:" + wn_child, wn_child, "WordNet", wn_child, wn.get(wn_child, "")),
        )
        for node_id, name, source, identifier, definition in node_specs:
            if node_id not in nodes:
                nodes[node_id] = {
                    "id": node_id, "name": name, "category": 0 if source == "MeSH" else 1,
                    "symbolSize": 24, "value": identifier, "source": source, "identifier": identifier,
                    "definition": definition.strip(), "rows": [],
                }
            if index not in nodes[node_id]["rows"]:
                nodes[node_id]["rows"].append(index)

        mp, mc, wp, wc = (spec[0] for spec in node_specs)
        edge_specs = (
            ("mesh_hierarchy", mp, mc, "MeSH 父子关系", "solid", "#579775", None),
            ("wn_hierarchy", wp, wc, "WordNet 父子关系", "solid", "#628fc4", None),
            ("candidate", mc, wc, "历史候选对应（待核查）", "dashed", "#9566c8", None),
            ("fuzzy", mp, wp, "父概念模糊匹配（待核查）", "dotted", "#d69a34", ratio),
        )
        for kind, source, target, name, line_type, color, value in edge_specs:
            key = kind, source, target
            if key in links:
                if kind == "fuzzy" and links[key]["value"] != value:
                    raise ResultsError("同一父概念配对出现不同的模糊匹配值")
                continue
            edge = {
                "source": source, "target": target, "name": name, "kind": kind,
                "lineStyle": {"type": line_type, "color": color, "width": 2 if kind.endswith("hierarchy") else 1.5},
                "symbol": ["none", "arrow"] if kind.endswith("hierarchy") else ["none", "none"],
            }
            if value is not None:
                edge["value"] = value
                edge["label"] = {"show": show_labels, "formatter": str(value), "color": color}
            links[key] = edge
    return list(nodes.values()), list(links.values())
