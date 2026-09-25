"""Rebuild the two 2019 input tables from the archived MeSH XML and WordNet 3.0."""

import csv
import re
from pathlib import Path
from xml.etree import ElementTree

from nltk import data as nltk_data
from nltk.corpus import wordnet


ROOT = Path(__file__).resolve().parent
nltk_data.path.insert(0, str(ROOT / "nltk_data"))
OUT = ROOT / "reproduced"
OUT.mkdir(exist_ok=True)


def mesh_rows():
    source = ROOT / "source" / "fredesc2019.xml"
    for _, element in ElementTree.iterparse(str(source), events=("end",)):
        if element.tag != "DescriptorRecord":
            continue
        descriptor_name = element.findtext("DescriptorName/String") or ""
        bracket = re.search(r"\[([^]]+)\]", descriptor_name)
        name = bracket.group(1) if bracket else descriptor_name
        concept = element.find("ConceptList/Concept")
        relations = []
        for relation in element.findall(".//ConceptRelation"):
            relations.extend([relation.get("RelationName", ""), relation.findtext("Concept2UI") or ""])
        terms = []
        for term_list in element.findall(".//TermList"):
            term = term_list.find("Term")
            if term is not None:
                terms.extend([term.findtext("TermUI") or "", term.findtext("String") or ""])
        scope = element.findtext(".//ScopeNote")
        trees = [node.text or "" for node in element.findall("TreeNumberList/TreeNumber")]
        yield {
            "c_ui": concept.findtext("ConceptUI") if concept is not None else "",
            "d_name": name,
            "d_ui": element.findtext("DescriptorUI") or "",
            "rel": repr(relations),
            "scopeNote": scope if scope is not None else repr(["null"]),
            "tree": repr(trees if trees else ["null"]),
            "word_list": repr(terms),
        }
        element.clear()


def wordnet_rows():
    for synset in wordnet.all_synsets(pos="n"):
        yield {
            "name": synset.lemma_names()[0],
            "synset": str(synset),
            "hyponymes": str(synset.hyponyms()),
            "hypernyms": str(synset.hypernyms()),
            "definition": synset.definition(),
            "exemple": str(synset.examples()),
            "list_word": str(synset.lemma_names()),
        }


def write(name, fields, rows):
    count = 0
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    print("{}: {} rows".format(name, count))


def read(name, directory, key):
    with (ROOT / directory / name).open(newline="", encoding="utf-8-sig") as stream:
        return {row[key]: row for row in csv.DictReader(stream)}


if __name__ == "__main__":
    write("mesh_from_xml.csv", ["c_ui", "d_name", "d_ui", "rel", "scopeNote", "tree", "word_list"], mesh_rows())
    write("wordnet_from_dict.csv", ["name", "synset", "hyponymes", "hypernyms", "definition", "exemple", "list_word"], wordnet_rows())
    archived_mesh = read("data_complet_MeSH(V3).csv", "data", "d_ui")
    rebuilt_mesh = read("mesh_from_xml.csv", "reproduced", "d_ui")
    assert archived_mesh == rebuilt_mesh
    print("MeSH extraction matches the archived table exactly")

    archived_wn = read("data_complet_WN(v3).csv", "data", "synset")
    rebuilt_wn = read("wordnet_from_dict.csv", "reproduced", "synset")
    assert archived_wn.keys() == rebuilt_wn.keys()
    differences = {"definition": 0, "exemple": 0}
    for key, row in archived_wn.items():
        rebuilt = rebuilt_wn[key]
        assert row["name"] == rebuilt["name"]
        for field in ["hyponymes", "hypernyms", "list_word"]:
            assert row[field] == rebuilt[field]
        for field in differences:
            differences[field] += row[field] != rebuilt[field]
    print("WordNet structure matches; text differences: {}".format(differences))
