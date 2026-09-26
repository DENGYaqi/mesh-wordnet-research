"""Independent, read-only reranking of the archived candidate pairs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding
from huggingface_hub import snapshot_download
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "runs" / "20260926T052631818279Z"
MODEL_DIR = ROOT / "models" / "semantic-minilm-l6-v2"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Pinned ONNX conversion of the Apache-2.0 model; never fetch a moving branch.
MODEL_REPO = "qdrant/all-MiniLM-L6-v2-onnx"
MODEL_REVISION = "5f1b8cd78bc4fb444dd171e59b18f3a3af89a079"
MODEL_SHA256 = "bbd7b466f6d58e646fdc2bd5fd67b2f5e93c0b687011bd4548c420f7bd46f0c5"
TOKENIZER_SHA256 = "da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0"
MODEL_FILES = ("model.onnx", "tokenizer.json", "tokenizer_config.json", "config.json")
INPUT_COLUMNS = (
    "MeSH_UI", "Name_MeSH", "Name_WN", "WN_synset", "def_MeSH", "def_WN"
)
GROUPS = ("ordinary_same_name", "changed_name", "ambiguous_same_name")
LABELS = (
    "等价", "MeSH 更宽", "WordNet 更宽", "相关但不等价", "无关", "不确定"
)
SEED = 42
SAMPLE_PER_GROUP = 40
# This pinned ONNX conversion's tokenizer_config.json sets max_length=128.
MAX_TOKENS = 128


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_candidates(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"缺少候选输入：{path}")
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(INPUT_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"候选输入缺少字段：{sorted(missing)}")
        rows = []
        for source_row, row in enumerate(reader, 1):
            if any(not row[column].strip() for column in INPUT_COLUMNS):
                raise ValueError(f"候选第 {source_row} 行缺少标识、名称或定义")
            rows.append({"source_row": source_row, **row})
    if len(rows) != 3018:
        raise ValueError(f"预期 3,018 对候选，实际 {len(rows)} 对；已停止评分")
    return rows


def wordnet_name_counts(path: Path) -> Counter:
    if not path.is_file():
        raise FileNotFoundError(f"缺少 WordNet 名词表，不能分层抽样：{path}")
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"name", "synset"}.issubset(reader.fieldnames or []):
            raise ValueError("WordNet 名词表缺少 name 或 synset 字段")
        senses = {}
        for row in reader:
            senses.setdefault(row["name"].casefold(), set()).add(row["synset"])
    return Counter({name: len(synsets) for name, synsets in senses.items()})


def make_review_sample(rows: list[dict], counts: Counter, path: Path) -> dict:
    pools = {group: [] for group in GROUPS}
    for row in rows:
        same = row["Name_MeSH"].casefold() == row["Name_WN"].casefold()
        group = (
            "changed_name" if not same else
            "ambiguous_same_name" if counts[row["Name_WN"].casefold()] > 1 else
            "ordinary_same_name"
        )
        pools[group].append(row)
    if any(len(pool) < SAMPLE_PER_GROUP for pool in pools.values()):
        raise ValueError(f"抽样组不足 40 条：{ {key: len(value) for key, value in pools.items()} }")
    rng = random.Random(SEED)
    selected = []
    for group in GROUPS:
        pool = pools[group]
        if group == "ambiguous_same_name":
            # Predeclared illustrative sense collision; chosen without looking at scores.
            air = next((row for row in pool if row["Name_MeSH"].casefold() == "air"), None)
            if air is not None:
                selected.append((group, air))
                pool = [row for row in pool if row is not air]
        selected.extend((group, row) for row in rng.sample(pool, SAMPLE_PER_GROUP - (group == "ambiguous_same_name" and air is not None)))
    selected.sort(key=lambda item: item[1]["source_row"])
    fields = ["source_row", "sample_group", *INPUT_COLUMNS, "review_relation", "review_note"]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for group, row in selected:
            writer.writerow({key: ({"sample_group": group, "review_relation": "", "review_note": ""}.get(key, row.get(key, ""))) for key in fields})
    return {"seed": SEED, "sample_per_group": SAMPLE_PER_GROUP,
            "pool_counts": {key: len(value) for key, value in pools.items()},
            "review_rows": len(selected), "air_included": any(row["Name_MeSH"].casefold() == "air" for _, row in selected),
            "allowed_labels": LABELS}


def ensure_model() -> tuple[Path, dict]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    marker = MODEL_DIR / "pinned_revision.json"
    if not (marker.is_file() and all((MODEL_DIR / name).is_file() for name in MODEL_FILES)):
        snapshot_download(
            repo_id=MODEL_REPO,
            revision=MODEL_REVISION,
            local_dir=str(MODEL_DIR),
            allow_patterns=[*MODEL_FILES, "vocab.txt", "special_tokens_map.json"],
        )
        write_json(marker, {"repo": MODEL_REPO, "revision": MODEL_REVISION})
    metadata = json.loads(marker.read_text(encoding="utf-8"))
    if metadata != {"repo": MODEL_REPO, "revision": MODEL_REVISION}:
        raise ValueError("本地模型版本标记与固定版本不一致")
    if any(not (MODEL_DIR / name).is_file() for name in MODEL_FILES):
        raise FileNotFoundError("固定模型下载不完整")
    onnx_sha = sha256(MODEL_DIR / "model.onnx")
    tokenizer_sha = sha256(MODEL_DIR / "tokenizer.json")
    if (onnx_sha, tokenizer_sha) != (MODEL_SHA256, TOKENIZER_SHA256):
        raise ValueError("本地 ONNX 模型或 tokenizer 校验值不匹配固定版本")
    return MODEL_DIR, {"name": MODEL_NAME, "repository": MODEL_REPO,
                       "revision": MODEL_REVISION, "onnx_sha256": onnx_sha,
                       "tokenizer_sha256": tokenizer_sha}


def token_overflows(texts: list[str], model_dir: Path) -> list[int]:
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    tokenizer.no_truncation()
    return [index for index, text in enumerate(texts) if len(tokenizer.encode(text).ids) > MAX_TOKENS]


def score(args: argparse.Namespace) -> None:
    output = Path(args.output_dir) if args.output_dir else ROOT / "runs" / ("semantic-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "manifest.json"
    candidate_file = Path(args.input_run) / "candidate_scores.csv"
    wordnet_file = ROOT / "data" / "data_complet_WN(v3).csv"
    manifest = {"status": "started", "method": "one shared definition encoder; cosine similarity; TF-IDF baseline",
                "tfidf": {"lowercase": True, "ngram_range": [1, 2], "norm": "l2"},
                "input_run": str(Path(args.input_run).resolve()), "candidate_file": str(candidate_file.resolve()),
                "candidate_sha256": sha256(candidate_file) if candidate_file.is_file() else None,
                "wordnet_file": str(wordnet_file.resolve()),
                "wordnet_sha256": sha256(wordnet_file) if wordnet_file.is_file() else None,
                "python": sys.version.split()[0], "seed": SEED, "max_model_tokens": MAX_TOKENS}
    write_json(manifest_path, manifest)
    try:
        rows = read_candidates(candidate_file)
        counts = wordnet_name_counts(wordnet_file)
        manifest["review_sample"] = make_review_sample(rows, counts, output / "review_sample.csv")
        manifest["status"] = "sampled_before_scoring"
        write_json(manifest_path, manifest)
        model_dir, model_metadata = ensure_model()
        manifest["model"] = model_metadata
        mesh_texts = [row["def_MeSH"].strip() for row in rows]
        wn_texts = [row["def_WN"].strip() for row in rows]
        overflows_mesh = token_overflows(mesh_texts, model_dir)
        overflows_wn = token_overflows(wn_texts, model_dir)
        manifest["truncation"] = {"mesh_source_rows": [i + 1 for i in overflows_mesh],
                                  "wordnet_source_rows": [i + 1 for i in overflows_wn]}
        manifest["status"] = "model_ready"
        write_json(manifest_path, manifest)

        encoder = TextEmbedding(model_name=MODEL_NAME, specific_model_path=str(model_dir), threads=1)
        mesh_vec = np.asarray(list(encoder.embed(mesh_texts, batch_size=64)), dtype=np.float64)
        wn_vec = np.asarray(list(encoder.embed(wn_texts, batch_size=64)), dtype=np.float64)
        if mesh_vec.shape != (3018, 384) or wn_vec.shape != (3018, 384):
            raise ValueError(f"模型返回意外的向量维度：{mesh_vec.shape} / {wn_vec.shape}")
        mesh_vec /= np.linalg.norm(mesh_vec, axis=1, keepdims=True)
        wn_vec /= np.linalg.norm(wn_vec, axis=1, keepdims=True)
        semantic = np.einsum("ij,ij->i", mesh_vec, wn_vec)
        tfidf = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, norm="l2", dtype=np.float64)
        matrix = tfidf.fit_transform(mesh_texts + wn_texts)
        baseline = np.asarray(matrix[:len(rows)].multiply(matrix[len(rows):]).sum(axis=1)).ravel()
        if not (np.isfinite(semantic).all() and np.isfinite(baseline).all()):
            raise ValueError("发现非有限分数，已停止输出")
        if np.ptp(semantic) < 1e-4:
            raise ValueError("语义分数没有足够区分度，已停止输出")

        ranked = sorted(range(len(rows)), key=lambda i: (-float(semantic[i]), rows[i]["source_row"]))
        fields = ["rank", "source_row", *INPUT_COLUMNS, "semantic_similarity", "tfidf_similarity", "model_truncated_mesh", "model_truncated_wordnet", "review_status"]
        with (output / "ranked_candidates.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for rank, index in enumerate(ranked, 1):
                writer.writerow({"rank": rank, **{key: rows[index][key] for key in ("source_row", *INPUT_COLUMNS)},
                                 "semantic_similarity": f"{semantic[index]:.9f}",
                                 "tfidf_similarity": f"{baseline[index]:.9f}",
                                 "model_truncated_mesh": index in overflows_mesh,
                                 "model_truncated_wordnet": index in overflows_wn,
                                 "review_status": "待核查"})
        manifest["status"] = "complete"
        manifest["pair_count"] = len(rows)
        manifest["semantic_score_range"] = [float(np.min(semantic)), float(np.max(semantic))]
        manifest["tfidf_score_range"] = [float(np.min(baseline)), float(np.max(baseline))]
        manifest["dependencies"] = {name: importlib.metadata.version(name) for name in
                                    ("fastembed", "huggingface-hub", "numpy", "onnxruntime", "scikit-learn", "scipy", "tokenizers")}
        manifest["ranked_sha256"] = sha256(output / "ranked_candidates.csv")
        manifest["review_sample_sha256"] = sha256(output / "review_sample.csv")
        write_json(manifest_path, manifest)
        print(f"完成：{len(rows)} 对候选；结果在 {output}")
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        write_json(manifest_path, manifest)
        raise


def evaluate(args: argparse.Namespace) -> None:
    run = Path(args.run_dir)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("评分运行尚未完成")
    with (run / "review_sample.csv").open(encoding="utf-8-sig", newline="") as stream:
        reviews = list(csv.DictReader(stream))
    if len(reviews) != 120:
        raise ValueError("审核表必须保持 120 行")
    if Counter(row["sample_group"] for row in reviews) != Counter({group: 40 for group in GROUPS}):
        raise ValueError("审核表的三个预抽样组必须各保留 40 行")
    invalid = [(row["source_row"], row["review_relation"]) for row in reviews
               if row["review_relation"].strip() not in LABELS]
    if invalid:
        raise ValueError(f"请先完成 120 条人工标注；空值或无效关系：{invalid[:5]}")
    with (run / "ranked_candidates.csv").open(encoding="utf-8-sig", newline="") as stream:
        ranked = {row["source_row"]: row for row in csv.DictReader(stream)}
    if len(set(row["source_row"] for row in reviews)) != 120 or any(row["source_row"] not in ranked for row in reviews):
        raise ValueError("审核表行号重复或不在评分结果中")
    if any(any(row[column] != ranked[row["source_row"]][column] for column in INPUT_COLUMNS) for row in reviews):
        raise ValueError("审核表中的标识、名称或定义已改动；请只填写审核列")
    report = {"review_rows": 120, "uncertain_excluded": sum(row["review_relation"].strip() == "不确定" for row in reviews),
              "positive_label": "等价", "metrics": {}, "note": "仅为人工审核样本上的排序表现，不是全量准确率。"}
    for group in ("all", *GROUPS):
        subset = [row for row in reviews if (group == "all" or row["sample_group"] == group)
                  and row["review_relation"].strip() != "不确定"]
        truth = [row["review_relation"].strip() == "等价" for row in subset]
        result = {"labeled": len(subset), "equivalent": sum(truth)}
        for method, column in (("semantic", "semantic_similarity"), ("tfidf", "tfidf_similarity")):
            result[f"{method}_average_precision"] = (float(average_precision_score(truth, [float(ranked[row["source_row"]][column]) for row in subset]))
                                                      if 0 < sum(truth) < len(truth) else None)
        report["metrics"][group] = result
    negatives = [row for row in reviews if row["review_relation"].strip() not in ("等价", "不确定")]
    positives = [row for row in reviews if row["review_relation"].strip() == "等价"]
    report["high_scoring_non_equivalent"] = [
        {"source_row": row["source_row"], "mesh": row["Name_MeSH"], "wordnet": row["Name_WN"],
         "relation": row["review_relation"], "semantic_similarity": ranked[row["source_row"]]["semantic_similarity"]}
        for row in sorted(negatives, key=lambda row: -float(ranked[row["source_row"]]["semantic_similarity"]))[:10]
    ]
    report["low_scoring_equivalent"] = [
        {"source_row": row["source_row"], "mesh": row["Name_MeSH"], "wordnet": row["Name_WN"],
         "semantic_similarity": ranked[row["source_row"]]["semantic_similarity"]}
        for row in sorted(positives, key=lambda row: float(ranked[row["source_row"]]["semantic_similarity"]))[:10]
    ]
    report["improved_over_tfidf_on_sample"] = (
        report["metrics"]["all"]["semantic_average_precision"] is not None and
        report["metrics"]["all"]["semantic_average_precision"] > report["metrics"]["all"]["tfidf_average_precision"]
    )
    write_json(run / "review_evaluation.json", report)
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    scorer = sub.add_parser("score", help="sample first, then score existing candidates")
    scorer.add_argument("--input-run", default=str(DEFAULT_INPUT))
    scorer.add_argument("--output-dir")
    evaluator = sub.add_parser("evaluate", help="evaluate only after all review labels are filled")
    evaluator.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    if args.command == "score":
        score(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
