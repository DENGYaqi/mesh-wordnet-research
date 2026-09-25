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
