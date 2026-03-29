from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from svm_studio.episode_mining import EpisodeDataset, mine_episodes
from svm_studio.itemset_mining import mine_itemsets


class MiningTests(unittest.TestCase):
    def test_apriori_mines_expected_itemset(self) -> None:
        transactions = [
            frozenset({"bread", "milk"}),
            frozenset({"bread", "diaper", "beer", "milk"}),
            frozenset({"milk", "diaper", "beer", "cola"}),
            frozenset({"bread", "milk", "diaper", "beer"}),
        ]

        results = mine_itemsets(
            transactions=transactions,
            dataset_name="demo",
            level="simple",
            min_support=0.50,
        )
        lookup = {result.items: result.support for result in results}

        self.assertIn(("beer", "diaper"), lookup)
        self.assertAlmostEqual(lookup[("beer", "diaper")], 0.75)
        self.assertIn(("bread", "milk"), lookup)
        self.assertAlmostEqual(lookup[("bread", "milk")], 0.75)

    def test_episode_mining_respects_support(self) -> None:
        dataset = EpisodeDataset(
            name="demo",
            level="simple",
            description="test",
            min_support=2 / 3,
            max_span=3,
            sequences=[
                ["A", "B", "C"],
                ["A", "B", "D"],
                ["A", "C", "D"],
            ],
        )

        results = mine_episodes(dataset)
        lookup = {result.events: result.support for result in results}

        self.assertIn(("A", "B"), lookup)
        self.assertAlmostEqual(lookup[("A", "B")], 2 / 3)
        self.assertNotIn(("A", "B", "C"), lookup)


if __name__ == "__main__":
    unittest.main()
