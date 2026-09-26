import shutil
import tempfile
import unittest
from pathlib import Path

from results import (
    ResultsError, build_neighbor_graph, load_results, local_neighbor_rows,
    search_candidates, search_neighbors,
)


RUN_DIR = Path("/runs/20260926T052631818279Z")


class ResultsTest(unittest.TestCase):
    def test_verified_run_and_search(self):
        manifest, candidates, neighbors = load_results(RUN_DIR)
        self.assertEqual((manifest["pairs"], len(candidates), len(neighbors)), (3018, 3018, 167))
        self.assertEqual(len(search_candidates(candidates, "  ")), 3018)
        self.assertEqual(search_candidates(candidates, "M0000003").iloc[0]["Name_MeSH"], "Abattoirs")
        self.assertTrue(search_candidates(candidates, "not-a-real-concept-xyz").empty)

    def test_incomplete_or_wrong_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            temp = Path(name)
            with self.assertRaisesRegex(ResultsError, "缺少"):
                load_results(temp)
            for filename in ("manifest.json", "candidate_scores.csv", "neighbor_matches.csv"):
                shutil.copyfile(RUN_DIR / filename, temp / filename)
            candidate_path = temp / "candidate_scores.csv"
            content = candidate_path.read_text(encoding="utf-8")
            candidate_path.write_text(content.replace("MeSH_UI", "wrong_id", 1), encoding="utf-8")
            with self.assertRaisesRegex(ResultsError, "缺少字段"):
                load_results(temp)

    def test_neighbor_graph_matches_archived_relations(self):
        _, candidates, neighbors = load_results(RUN_DIR)
        self.assertEqual(len(search_neighbors(neighbors, candidates, "vegetarian.n.01")), 1)
        self.assertEqual(len(search_neighbors(neighbors, candidates, "no-such-synset")), 0)
        self.assertEqual(len(local_neighbor_rows(neighbors, 0)), 11)

        nodes, links = build_neighbor_graph(candidates, neighbors, show_labels=False)
        self.assertEqual((len(nodes), len(links)), (530, 618))
        self.assertEqual({node["source"] for node in nodes}, {"MeSH", "WordNet"})
        self.assertEqual(
            {link["kind"] for link in links},
            {"mesh_hierarchy", "wn_hierarchy", "candidate", "fuzzy"},
        )
        person = next(link for link in links if link["kind"] == "fuzzy" and
                      link["target"] == "wn:person.n.01" and
                      next(node for node in nodes if node["id"] == link["source"])["name"] == "Persons")
        self.assertEqual(person["value"], 83)
        self.assertEqual(person["symbol"], ["none", "none"])
        self.assertTrue(any(link["kind"] == "mesh_hierarchy" and
                            next(node for node in nodes if node["id"] == link["target"])["name"] == "Vegetarians"
                            for link in links))
        self.assertTrue(any(link["kind"] == "candidate" and link["target"] == "wn:vegetarian.n.01" and
                            next(node for node in nodes if node["id"] == link["source"])["name"] == "Vegetarians"
                            for link in links))
        archived_parent = next(node for node in nodes if node["name"] == "Elementary Particles")
        self.assertEqual((archived_parent["identifier"], archived_parent["definition"]), ("", ""))


if __name__ == "__main__":
    unittest.main()
