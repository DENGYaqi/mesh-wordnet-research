"""Replay the archived two-vector SIF comparison using saved word vectors.

The original word-frequency table was not saved. Model vocabulary counts are used
as weights here; they are sufficient to test the two-vector PCA failure mode,
but this is not a byte-for-byte replay of the original training run.
"""

import ast
import csv
from pathlib import Path

import numpy as np
from gensim.models import KeyedVectors


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "data_using_sim_sif_with_concept_a_verifie.csv"
OUT = ROOT / "reproduced" / "sif_from_saved_vectors.csv"


def sentence_vector(tokens, model, total_count):
    tokens = [token for token in tokens if token in model.vocab]
    if not tokens:
        tokens = ["building"]  # The fallback in the 2019 notebook.
    weights = [0.001 / (0.001 + model.vocab[token].count / total_count) for token in tokens]
    return np.average([model[token] for token in tokens], axis=0, weights=weights)


def original_two_vector_sif(first, second):
    vectors = np.array([first, second])
    _, _, right = np.linalg.svd(vectors, full_matrices=False)
    pc = right[0]
    residuals = vectors - np.outer(vectors @ pc, pc)
    denominator = np.linalg.norm(residuals[0]) * np.linalg.norm(residuals[1])
    assert denominator > 1e-15, "Degenerate zero residual"
    return abs(float(residuals[0] @ residuals[1] / denominator))


def main():
    mesh = KeyedVectors.load(str(ROOT / "models" / "wordvector_MeSH.kv"), mmap="r")
    wn = KeyedVectors.load(str(ROOT / "models" / "wordvector_WN.kv"), mmap="r")
    assert mesh.vector_size == wn.vector_size == 300
    mesh_total = sum(item.count for item in mesh.vocab.values())
    wn_total = sum(item.count for item in wn.vocab.values())
    rows = []
    with SOURCE.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            left = sentence_vector(ast.literal_eval(row["word_list_def_MeSH"]), mesh, mesh_total)
            right = sentence_vector(ast.literal_eval(row["word_list_def_WN"]), wn, wn_total)
            score = original_two_vector_sif(left, right)
            saved_score = ast.literal_eval(row["Similarity_highest"])[0]
            assert abs(score - 1) < 1e-8 and abs(score - saved_score) < 1e-8
            rows.append({"Name_MeSH": row["Name_MeSH"], "Name_WN": row["Name_WN"], "SIF_score": score})
    assert len(rows) == 3018
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["Name_MeSH", "Name_WN", "SIF_score"])
        writer.writeheader()
        writer.writerows(rows)
    print("Loaded archived 300-dimensional MeSH and WordNet vectors")
    print("Recomputed {} two-vector SIF comparisons; every absolute score is approximately 1".format(len(rows)))
    print("Historical word frequencies and model training were not replayed")


if __name__ == "__main__":
    main()
