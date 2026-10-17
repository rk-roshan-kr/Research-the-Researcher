import unittest
from pydantic import ValidationError
from academic_pipeline.local_llm import ExtractedClaimDetail, PaperClaimsExtraction

class TestClaimExtraction(unittest.TestCase):
    def test_extracted_claim_detail_validation(self):
        # Valid claim detail
        detail = ExtractedClaimDetail(
            claim="Curriculum learning improves learning speed",
            method="CKA",
            dataset="CIFAR10",
            metric="accuracy",
            confidence=0.95
        )
        self.assertEqual(detail.claim, "Curriculum learning improves learning speed")
        self.assertEqual(detail.method, "CKA")
        self.assertEqual(detail.dataset, "CIFAR10")
        self.assertEqual(detail.metric, "accuracy")
        self.assertEqual(detail.confidence, 0.95)

        # Invalid confidence
        with self.assertRaises(ValidationError):
            ExtractedClaimDetail(
                claim="Test",
                method="N/A",
                dataset="N/A",
                metric="N/A",
                confidence=1.5  # Should be <= 1.0
            )

    def test_paper_claims_extraction_validation(self):
        data = {
            "claims": [
                {
                    "claim": "Curriculum learning works",
                    "method": "CKA",
                    "dataset": "CEBRA",
                    "metric": "similarity",
                    "confidence": 0.9
                }
            ],
            "open_questions": ["Does it scale?"],
            "methods": ["CKA"],
            "datasets": ["CEBRA"],
            "metrics": ["similarity"]
        }
        extracted = PaperClaimsExtraction(**data)
        self.assertEqual(len(extracted.claims), 1)
        self.assertEqual(extracted.claims[0].method, "CKA")
        self.assertEqual(extracted.open_questions, ["Does it scale?"])

if __name__ == "__main__":
    unittest.main()
