"""Run the archived 2019 MeSH–WordNet method without editing its source files.

``verify`` checks the preserved results. ``all`` also retrains the two
Word2Vec models and writes a separate, timestamped run directory. The
historical SIF calculation is intentionally retained, including its known
all-one-score failure mode.
"""

import argparse
import ast
import csv
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import gensim
import nltk
import numpy as np
import spacy
from fuzzywuzzy import fuzz
from gensim.models import Word2Vec
from nltk import data as nltk_data
from nltk.corpus import stopwords, wordnet
from sklearn.decomposition import TruncatedSVD

from reproduce import read_rows, replay_pairs


ROOT = Path(__file__).resolve().parent
nltk_data.path.insert(0, str(ROOT / "nltk_data"))
SEED = 42
FIELDS = ["MeSH_UI", "Name_MeSH", "Name_WN", "WN_synset", "def_MeSH", "def_WN", "Similarity_highest"]
NEIGHBOR_FIELDS = [
    "synset_father_wn", "Parti_Ratio(fuzz)_between_father_concept",
    "synset_child_wn", "identify_father_MeSH", "identify_child_MesH",
]


def invoke(script, *args):
    subprocess.run([sys.executable, str(ROOT / script), *args], check=True, cwd=str(ROOT))


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_training_sentences(rows, field, nlp):
    # Notebook cells 9–15: clean definitions, drop missing/duplicates, split.
    texts = (re.sub("[^A-Za-z']+", " ", row[field]).lower() for row in rows)
    seen = set()
    sentences = []
    for doc in nlp.pipe(texts, batch_size=5000):
        lemmas = [token.lemma_ for token in doc if not token.is_stop]
        if len(lemmas) <= 2:
            continue
        cleaned = " ".join(lemmas)
        if cleaned not in seen:
            seen.add(cleaned)
            sentences.append(cleaned.split())
    sentences.append(["UNK"])  # Historical out-of-vocabulary sentinel.
    return sentences


def train_one(sentences, path):
    # Notebook cells 24–31, with a fixed seed and one worker for repeatability.
    model = Word2Vec(
        min_count=20, window=2, vector_size=300, sample=6e-5,
        alpha=0.03, min_alpha=0.0007, negative=20,
        workers=1, hs=1, sg=1, seed=SEED,
    )
    model.build_vocab(sentences)
    assert model.corpus_count == len(sentences) and len(model.wv)
    model.train(sentences, total_examples=model.corpus_count, epochs=30)
    model.save(str(path))
    model.wv.save(str(path.with_suffix(".kv")))
    loaded = Word2Vec.load(str(path))
    assert loaded.vector_size == 300 and len(loaded.wv) == len(model.wv)
    return loaded


def sentence_vector(text, vectors, freqs, total, stop):
    tokens = [token.lower() for token in nltk.word_tokenize(text)]
    tokens = [token for token in tokens if token not in stop and token in vectors]
    if not tokens:
        tokens = ["building"]  # Notebook cell 46 fallback.
    weights = [0.001 / (0.001 + freqs.get(token, 0) / total) for token in tokens]
    return np.average([vectors[token] for token in tokens], axis=0, weights=weights)


def historical_sif(first, second):
    # Notebook cell 61 calls this separately for each pair. Keep that behavior.
    embeddings = np.asarray([first, second])
    pc = TruncatedSVD(n_components=1, n_iter=7, random_state=0).fit(embeddings).components_[0]
    residuals = embeddings - np.outer(embeddings.dot(pc), pc)
    denominator = np.linalg.norm(residuals[0]) * np.linalg.norm(residuals[1])
    if denominator == 0:
        raise ValueError("SIF produced a zero residual")
    return abs(float(residuals[0].dot(residuals[1]) / denominator))


def score_candidates(mesh_model, wn_model, mesh_freqs, wn_freqs, out):
    pairs = replay_pairs()
    archived = read_rows("data_using_sim_sif_with_concept_a_verifie.csv")
    assert len(pairs) == len(archived) == 3018
    mesh_def = {row["d_name"]: row["scopeNote"] for row in read_rows("data_MeSH_concept_verifie(V1).csv")}
    wn_def = {row["c_name"]: row["definition"] for row in read_rows("data_wn_concept_verifie(V1).csv")}
    stop = set(stopwords.words("english"))
    mesh_total = sum(mesh_freqs.values())
    wn_total = sum(wn_freqs.values())
    results = []
    for pair, saved in zip(pairs, archived):
        assert (pair["Name_MeSH"], pair["Name_WN"]) == (saved["Name_MeSH"], saved["Name_WN"])
        left = mesh_def[pair["Name_MeSH"]]
        right = wn_def[pair["Name_WN"]]
        embedding1 = sentence_vector(left, mesh_model.wv, mesh_freqs, mesh_total, stop)
        embedding2 = sentence_vector(right, wn_model.wv, wn_freqs, wn_total, stop)
        score = historical_sif(embedding1, embedding2)
        if not np.isfinite(score):
            raise ValueError("Non-finite SIF score for " + pair["Name_MeSH"])
        results.append(dict(pair, def_MeSH=left, def_WN=right, Similarity_highest=repr([score])))
    write_csv(out / "candidate_scores.csv", FIELDS, results)
    return len(results), sum(abs(ast.literal_eval(row["Similarity_highest"])[0] - 1) < 1e-8 for row in results)


def recompute_partition(out):
    mesh_rows = read_rows("data_MeSH_searche_by_data_using_sim_sif_with_concept_a_verifie.csv")
    wn_rows = read_rows("data_WN_searche_by_data_using_sim_sif_with_concept_a_verifie.csv")
    assert len(mesh_rows) == len(wn_rows) == 3018
    partitions = {False: ([], []), True: ([], [])}
    invalid = []
    for mesh_row, wn_row in zip(mesh_rows, wn_rows):
        match = re.fullmatch(r"Synset\('(.+)'\)", wn_row["synset"])
        if not match:
            invalid.append((mesh_row["d_name"], wn_row["synset"]))
            continue
        has_hyponyms = bool(wordnet.synset(match.group(1)).hyponyms())
        partitions[has_hyponyms][0].append(mesh_row)
        partitions[has_hyponyms][1].append(wn_row)
    assert invalid == [("Names", "synset")]
    for has_hyponyms, mesh_name, wn_name in (
        (False, "mesh_no_hypo.csv", "wn_no_hypo.csv"),
        (True, "mesh_has_hypo.csv", "wn_has_hypo.csv"),
    ):
        mesh_part, wn_part = partitions[has_hyponyms]
        assert mesh_part == read_rows(mesh_name) and wn_part == read_rows(wn_name)
        write_csv(out / mesh_name, list(mesh_rows[0]), mesh_part)
        write_csv(out / wn_name, list(wn_rows[0]), wn_part)
    return partitions[True]


def recompute_neighbors(out):
    # The archived Notebook uses the first MeSH tree number and follows the
    # first WordNet hypernym branch. This reproduces its saved 167 rows.
    mesh_rows = read_rows("data_complet_MeSH(V3).csv")
    by_tree = {
        tree: row["d_name"]
        for row in mesh_rows
        for tree in ast.literal_eval(row["tree"])
        if tree != "null"
    }
    mesh_has, wn_has = recompute_partition(out)
    assert len(mesh_has) == len(wn_has) == 716
    results = []
    for mesh_row, wn_row in zip(mesh_has, wn_has):
        trees = ast.literal_eval(mesh_row["tree"])
        if not trees or "." not in trees[0]:
            continue
        parent_name = by_tree.get(trees[0].rsplit(".", 1)[0])
        if not parent_name:
            continue
        match = re.fullmatch(r"Synset\('(.+)'\)", wn_row["synset"])
        if not match:
            raise ValueError("Malformed WordNet synset: " + wn_row["synset"])
        child = wordnet.synset(match.group(1))
        ancestors = sorted(child.hypernyms(), key=lambda item: item.name())
        visited = set()
        while ancestors:
            parent = ancestors[0]
            if parent in visited:
                break
            visited.add(parent)
            ratio = fuzz.partial_ratio(parent.name().split(".")[0], parent_name)
            if ratio >= 80:
                results.append({
                    "synset_father_wn": parent.name(),
                    "Parti_Ratio(fuzz)_between_father_concept": ratio,
                    "synset_child_wn": child.name(),
                    "identify_father_MeSH": parent_name,
                    "identify_child_MesH": mesh_row["d_name"],
                })
                break
            ancestors = sorted(parent.hypernyms(), key=lambda item: item.name())
    saved = read_rows("concept_fusonable_after_second_algo_for_concept_has_hypo.csv")
    normalize = lambda row: tuple(str(row[field]) for field in NEIGHBOR_FIELDS)
    assert len(results) == len(saved) == 167
    assert [normalize(row) for row in results] == [normalize(row) for row in saved]
    write_csv(out / "neighbor_matches.csv", NEIGHBOR_FIELDS, results)
    return len(results)


def verify():
    invoke("extract_sources.py")
    invoke("reproduce.py", "--raw")
    invoke("check_saved_sif.py")
    out = ROOT / "reproduced"
    count = recompute_neighbors(out)
    print("Archived neighbor algorithm reproduced: {} rows".format(count))


def run_all():
    verify()
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    run_dir = ROOT / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    print("Preprocessing archived MeSH definitions", flush=True)
    mesh_sentences = prepare_training_sentences(read_rows("data_complet_MeSH(V3).csv"), "scopeNote", nlp)
    print("Preprocessing archived WordNet definitions", flush=True)
    wn_sentences = prepare_training_sentences(read_rows("data_complet_WN(v3).csv"), "definition", nlp)
    mesh_freqs = Counter(token for sentence in mesh_sentences for token in sentence)
    wn_freqs = Counter(token for sentence in wn_sentences for token in sentence)
    print("Training MeSH Word2Vec: {} sentences".format(len(mesh_sentences)), flush=True)
    mesh_model = train_one(mesh_sentences, run_dir / "word2vec_mesh.model")
    print("Training WordNet Word2Vec: {} sentences".format(len(wn_sentences)), flush=True)
    wn_model = train_one(wn_sentences, run_dir / "word2vec_wordnet.model")
    pairs, all_one = score_candidates(mesh_model, wn_model, mesh_freqs, wn_freqs, run_dir)
    neighbors = recompute_neighbors(run_dir)
    manifest = {
        "python": sys.version.split()[0], "nltk": nltk.__version__,
        "spacy": spacy.__version__, "gensim": gensim.__version__,
        "seed": SEED, "workers": 1,
        "training_sentences": {"mesh": len(mesh_sentences), "wordnet": len(wn_sentences)},
        "vocabulary": {"mesh": len(mesh_model.wv), "wordnet": len(wn_model.wv)},
        "pairs": pairs, "scores_near_one": all_one, "neighbors": neighbors,
        "source_sha256": {
            name: checksum(ROOT / "data" / name)
            for name in ("data_complet_MeSH(V3).csv", "data_complet_WN(v3).csv",
                         "data_MeSH_concept_verifie(V1).csv", "data_wn_concept_verifie(V1).csv")
        },
        "limitations": [
            "The 2019 spaCy model version and random state were not archived; model values need not match.",
            "The archived WordNet verification subset cannot be rebuilt exactly from the surviving notebook.",
            "SIF removes a principal component separately for each pair and takes absolute cosine.",
            "The two Word2Vec spaces were trained separately and are not aligned.",
        ],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Historical run completed: {} pairs, {} scores near one, {} neighbors".format(pairs, all_one, neighbors))
    print("Run output: " + str(run_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "all"))
    arguments = parser.parse_args()
    if arguments.command == "verify":
        verify()
    else:
        run_all()
