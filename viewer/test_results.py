import shutil
import tempfile
import unittest
from pathlib import Path

from results import ResultsError, load_results, search_candidates


RUN_DIR = Path("/runs/20260924T161334509381Z")


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


if __name__ == "__main__":
    unittest.main()
