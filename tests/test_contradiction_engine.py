import unittest
from unittest.mock import MagicMock
from academic_pipeline.nli_filter import NLIFilter, NLIResult

class TestContradictionEngine(unittest.TestCase):
    def test_nli_filter_fallback_without_transformers(self):
        # We can construct the NLIFilter and test fallback
        nli = NLIFilter()
        # Mock _model_loaded to raise ImportError
        nli._model_loaded = MagicMock(side_effect=ImportError("mock error"))
        
        candidates = [
            {
                "claim_yours": "Curriculum learning helps representation structure",
                "claim_external": "Curriculum learning degrades representation structure",
                "source_external": "Chen et al. 2023",
                "similarity": 0.85
            }
        ]
        results = nli.filter_candidates(candidates)
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].passes_filter)
        self.assertEqual(results[0].p_contradiction, 0.40)
        self.assertIn("nli_unavailable", results[0].quality_flags)

    def test_find_contradiction_candidates_fallback(self):
        nli = NLIFilter()
        nli._model_loaded = MagicMock(side_effect=ImportError("mock error"))
        
        my_claims = [{"id": "c1", "claim": "Curriculum learning helps"}]
        their_claims = [{"id": "c2", "claim": "Curriculum learning degrades"}]
        
        # Should fall back to passing the candidates directly
        pairs = nli.find_contradiction_candidates(my_claims, their_claims, embedding_threshold=0.3)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][0]["id"], "c1")
        self.assertEqual(pairs[0][1]["id"], "c2")

if __name__ == "__main__":
    unittest.main()
