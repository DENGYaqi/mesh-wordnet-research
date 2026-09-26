"""Replay and audit the 2019 matching results without changing the archived CSVs."""

import argparse
import ast
import csv
from pathlib import Path

from nltk import data as nltk_data
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer


ROOT = Path(__file__).resolve().parent
nltk_data.path.insert(0, str(ROOT / "nltk_data"))
DATA = ROOT / "data"
OUT = ROOT / "reproduced"


def read_rows(name):
    with (DATA / name).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_rows(name, fieldnames, rows):
    OUT.mkdir(exist_ok=True)
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def replay_pairs():
    # This is the lookup in cells 55 and 58 of the historical SIF notebook.
    mesh = {row["d_name"]: row for row in read_rows("data_MeSH_concept_verifie(V1).csv")}
    wordnet = {row["c_name"]: row for row in read_rows("data_wn_concept_verifie(V1).csv")}
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    pairs = []
    for mesh_name, mesh_row in mesh.items():
        wordnet_name = mesh_name
        if wordnet_name not in wordnet:
            lemma = lemmatizer.lemmatize(mesh_name)
            # NLTK 3.4.5 returned the original casing for words of length <= 2.
            wordnet_name = lemma if len(lemma) <= 2 else stemmer.stem(lemma)
        if wordnet_name in wordnet:
            wordnet_row = wordnet[wordnet_name]
            pairs.append({
                "MeSH_UI": mesh_row["c_ui"],
                "Name_MeSH": mesh_name,
                "Name_WN": wordnet_name,
                "WN_synset": wordnet_row["synset"],
            })
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", action="store_true", help="also verify raw MeSH XML and WordNet noun data")
    args = parser.parse_args()

    pairs = replay_pairs()
    saved = read_rows("data_using_sim_sif_with_concept_a_verifie.csv")
    saved_pairs = [(row["Name_MeSH"], row["Name_WN"]) for row in saved]
    replayed_pairs = [(row["Name_MeSH"], row["Name_WN"]) for row in pairs]
    assert replayed_pairs == saved_pairs, "Replayed pairs differ from the historical CSV"

    enriched_wn = read_rows("data_WN_searche_by_data_using_sim_sif_with_concept_a_verifie.csv")
    lost_indexes = [i for i, row in enumerate(enriched_wn) if row["synset"] == "synset"]
    invalid = [pairs[i] for i in lost_indexes]
    valid = [row for i, row in enumerate(pairs) if i not in lost_indexes]
    assert len(pairs) == 3018 and len(invalid) == 1 and len(valid) == 3017
    assert invalid[0]["Name_MeSH"] == "Names" and invalid[0]["Name_WN"] == "name"

    no_hypo = read_rows("mesh_no_hypo.csv")
    has_hypo = read_rows("mesh_has_hypo.csv")
    neighbor = read_rows("concept_fusonable_after_second_algo_for_concept_has_hypo.csv")
    partition = {(row["c_ui"], row["d_name"]) for row in no_hypo + has_hypo}
    assert partition == {(row["MeSH_UI"], row["Name_MeSH"]) for row in valid}
    assert (len(no_hypo), len(has_hypo), len(neighbor)) == (2301, 716, 167)
    assert all(int(row["Parti_Ratio(fuzz)_between_father_concept"]) >= 80 for row in neighbor)

    scores = [ast.literal_eval(row["Similarity_highest"])[0] for row in saved]
    assert len(scores) == len(pairs) and all(abs(score - 1) < 1e-12 for score in scores)

    fields = ["MeSH_UI", "Name_MeSH", "Name_WN", "WN_synset"]
    write_rows("historical_pairs.csv", fields, pairs)
    write_rows("valid_pairs.csv", fields, valid)
    print("2019 pair lookup reproduced: 3018 rows, exact order and names")
    print("Enriched pairs: 3017; one valid pair was lost when WN enrichment selected its CSV header (Names -> name)")
    print("Partition: 2301 without hyponyms + 716 with hyponyms; neighbor matches: 167")
    print("Reported integrable count: 2301 + 167 = 2468")
    print("All 3018 saved SIF scores are effectively 1; they cannot rank these pairs")
    print("The report's 3019/717 counts include a CSV header read as data")

    if args.raw:
        from xml.etree import ElementTree
        import nltk.corpus
        from fuzzywuzzy import fuzz

        xml_ids = set()
        xml_path = ROOT / "source" / "fredesc2019.xml"
        for _, element in ElementTree.iterparse(str(xml_path), events=("end",)):
            if element.tag == "DescriptorRecord":
                xml_ids.add(element.findtext("DescriptorUI"))
                element.clear()
        mesh_ids = {row["c_ui"] for row in read_rows("data_complet_MeSH(V3).csv")}
        assert xml_ids == {row["d_ui"] for row in read_rows("data_complet_MeSH(V3).csv")}
        assert {row["MeSH_UI"] for row in valid} <= mesh_ids

        nouns = {str(synset) for synset in nltk.corpus.wordnet.all_synsets(pos="n")}
        saved_nouns = {row["synset"] for row in read_rows("data_complet_WN(v3).csv")}
        assert nouns == saved_nouns
        print("Raw sources verified: {} MeSH descriptors; {} WordNet noun synsets".format(len(xml_ids), len(nouns)))

        mesh_trees = {}
        for row in read_rows("data_complet_MeSH(V3).csv"):
            mesh_trees.setdefault(row["d_name"], []).extend(ast.literal_eval(row["tree"]))
        for row in neighbor:
            parent_trees = mesh_trees[row["identify_father_MeSH"]]
            child_trees = mesh_trees[row["identify_child_MesH"]]
            assert any(child.startswith(parent + ".") for child in child_trees for parent in parent_trees)
            parent = nltk.corpus.wordnet.synset(row["synset_father_wn"])
            child = nltk.corpus.wordnet.synset(row["synset_child_wn"])
            assert any(parent in path[:-1] for path in child.hypernym_paths())
            score = fuzz.partial_ratio(row["synset_father_wn"].split(".")[0], row["identify_father_MeSH"])
            assert score == int(row["Parti_Ratio(fuzz)_between_father_concept"])
        print("All 167 saved neighbor matches pass both raw hierarchies and the historical fuzzy score")


if __name__ == "__main__":
    main()
