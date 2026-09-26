"""Run from the repo root: uv run python -m unittest tests.test_hybrid_search"""

import random
import unittest
from itertools import chain

from core.services.retrieval_service import min_max_normalize, weighted_sum


def reference_hybrid(semantic: list[tuple[int, float]], bm25: list[tuple[int, float]], n: int, weights=(0.8, 0.2)):
    """The thesis' PublicationService._hybrid_search, on (id, raw score) lists."""

    def normalize(scores):
        low, high = min(scores), max(scores)
        if low == high:
            return [0.0 for _ in scores]
        return [(score - low) / (high - low) for score in scores]

    ids_semantic, scores_semantic = [i for i, _ in semantic], normalize([s for _, s in semantic])
    ids_bm25, scores_bm25 = [i for i, _ in bm25], normalize([s for _, s in bm25])
    scores_semantic = [weights[0] * score for score in scores_semantic]
    scores_bm25 = [weights[1] * score for score in scores_bm25]
    merged = {}
    for work, score in zip(chain(ids_semantic, ids_bm25), chain(scores_semantic, scores_bm25), strict=True):
        if work not in merged:
            merged[work] = score
        else:
            merged[work] += score
    return sorted(merged.items(), key=lambda x: x[1], reverse=True)[:n]


class HybridSearchTest(unittest.TestCase):
    def test_min_max_normalize(self):
        self.assertEqual(min_max_normalize({7: 3.0, 5: 2.0, 9: 1.0}), {7: 1.0, 5: 0.5, 9: 0.0})
        self.assertEqual(min_max_normalize({7: 2.0, 5: 2.0}), {7: 0.0, 5: 0.0})
        self.assertEqual(min_max_normalize({}), {})

    def test_weighted_sum_over_the_union(self):
        fused = weighted_sum([{1: 1.0, 2: 0.5}, {2: 1.0, 3: 0.5}], [0.8, 0.2])
        self.assertEqual(list(fused), [1, 2, 3])
        self.assertAlmostEqual(fused[2], 0.8 * 0.5 + 0.2 * 1.0)
        self.assertAlmostEqual(fused[3], 0.2 * 0.5)

    def test_same_ranking_and_scores_as_the_thesis_code(self):
        rng = random.Random(0)
        for trial in range(200):
            with self.subTest(trial=trial):
                ids = rng.sample(range(1000), 60)
                # overlapping result lists, with ties: scores rounded to one decimal
                semantic = sorted(((i, round(rng.uniform(0.2, 0.7), 1)) for i in ids[:40]), key=lambda x: -x[1])
                bm25 = sorted(((i, round(rng.uniform(0, 30), 1)) for i in ids[20:]), key=lambda x: -x[1])
                n = rng.choice([5, 20, 50])
                blended = weighted_sum([min_max_normalize(dict(semantic)), min_max_normalize(dict(bm25))], [0.8, 0.2])
                self.assertEqual(list(blended.items())[:n], reference_hybrid(semantic, bm25, n))


if __name__ == "__main__":
    unittest.main()
