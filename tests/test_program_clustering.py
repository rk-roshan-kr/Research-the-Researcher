import unittest
import numpy as np
from unittest.mock import MagicMock, patch

from academic_pipeline.university_discovery import UniversityDiscoveryEngine
from academic_pipeline.cache_manager import CacheManager
from academic_pipeline.investigation_engine import InvestigationEngine

class TestProgramClustering(unittest.TestCase):

    def setUp(self):
        import tempfile
        self.temp_dir = tempfile.TemporaryDirectory()
        # Create minimal mocked dependencies for UniversityDiscoveryEngine
        self.mock_cache = MagicMock(spec=CacheManager)
        self.mock_cache.cache_dir = self.temp_dir.name
        self.mock_inv = MagicMock(spec=InvestigationEngine)
        self.mock_llm = MagicMock()
        
        # Mock extract_claims_and_evidence for any engine calls
        async def mock_extract(*args, **kwargs):
            return {
                "claims": [
                    {
                        "claim": "Curriculum learning works",
                        "method": "CKA",
                        "dataset": "CEBRA",
                        "metric": "similarity",
                        "confidence": 0.9
                    }
                ],
                "open_questions": [],
                "methods": ["CKA"],
                "datasets": ["CEBRA"],
                "metrics": ["similarity"]
            }
        self.mock_llm.extract_claims_and_evidence.side_effect = mock_extract
        
        self.mock_llm.extract_claims_and_evidence_sync.return_value = {
            "claims": [],
            "open_questions": [],
            "methods": [],
            "datasets": [],
            "metrics": []
        }
        
        # Instantiate engine
        self.engine = UniversityDiscoveryEngine(
            cache_manager=self.mock_cache,
            investigation_engine=self.mock_inv,
            llm_client=self.mock_llm
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_extract_cluster_keywords(self):
        papers = [
            {"title": "continual learning of representation dynamics", "abstract": "we explore curriculum learning and continual learning"},
            {"title": "representation geometry in continual learning", "abstract": "continual learning representation representations"}
        ]
        
        keywords = self.engine._extract_cluster_keywords(papers)
        # Verify that common stop words are ignored and important keywords are extracted
        self.assertIn("continual", keywords)
        self.assertIn("learning", keywords)
        self.assertIn("representation", keywords)
        self.assertNotIn("we", keywords)
        self.assertNotIn("explore", keywords)

    @patch("sklearn.cluster.HDBSCAN")
    def test_cluster_papers_hdbscan_success(self, mock_hdbscan_class):
        # Mock HDBSCAN to return two non-noise clusters
        mock_instance = MagicMock()
        mock_instance.fit_predict.return_value = np.array([0, 0, 1, 1, 0, 1])
        mock_hdbscan_class.return_value = mock_instance
        
        embeddings = np.random.randn(6, 1024).astype(np.float32)
        papers = [{"title": f"Paper {i}"} for i in range(6)]
        
        labels = self.engine._cluster_papers(papers, embeddings)
        
        # HDBSCAN should be used successfully
        mock_hdbscan_class.assert_called_once()
        np.testing.assert_array_equal(labels, np.array([0, 0, 1, 1, 0, 1]))

    @patch("sklearn.cluster.HDBSCAN")
    @patch("sklearn.cluster.AgglomerativeClustering")
    def test_cluster_papers_hdbscan_fallback_to_agglomerative(self, mock_agg_class, mock_hdbscan_class):
        # Case A: HDBSCAN yields < 2 clusters (e.g. all noise or single cluster)
        mock_hdbscan_instance = MagicMock()
        mock_hdbscan_instance.fit_predict.return_value = np.array([-1, -1, 0, 0, 0, 0])
        mock_hdbscan_class.return_value = mock_hdbscan_instance
        
        mock_agg_instance = MagicMock()
        mock_agg_instance.fit_predict.return_value = np.array([0, 0, 1, 1, 2, 2])
        mock_agg_class.return_value = mock_agg_instance
        
        embeddings = np.random.randn(6, 1024).astype(np.float32)
        papers = [{"title": f"Paper {i}"} for i in range(6)]
        
        labels = self.engine._cluster_papers(papers, embeddings)
        
        # Verify HDBSCAN was called, yielded <2 non-noise clusters, and Agglomerative fallback took over
        mock_hdbscan_class.assert_called_once()
        mock_agg_class.assert_called_once_with(n_clusters=3, metric='euclidean', linkage='ward')
        np.testing.assert_array_equal(labels, np.array([0, 0, 1, 1, 2, 2]))

    @patch("sklearn.cluster.HDBSCAN")
    @patch("sklearn.cluster.AgglomerativeClustering")
    def test_cluster_papers_hdbscan_failure_fallback(self, mock_agg_class, mock_hdbscan_class):
        # Case B: HDBSCAN raises an exception (e.g. if library or parameters fail)
        mock_hdbscan_class.side_effect = ImportError("Simulated failure")
        
        mock_agg_instance = MagicMock()
        mock_agg_instance.fit_predict.return_value = np.array([0, 0, 1, 1])
        mock_agg_class.return_value = mock_agg_instance
        
        embeddings = np.random.randn(4, 1024).astype(np.float32)
        papers = [{"title": f"Paper {i}"} for i in range(4)]
        
        labels = self.engine._cluster_papers(papers, embeddings)
        
        # Agglomerative clustering fallback should run
        mock_agg_class.assert_called_once_with(n_clusters=3, metric='euclidean', linkage='ward')
        np.testing.assert_array_equal(labels, np.array([0, 0, 1, 1]))

    @patch("academic_pipeline.university_discovery.cached_encode")
    def test_extract_programs_for_dossier_flow(self, mock_cached_encode):
        # Mock cached_encode to return random vectors
        mock_cached_encode.return_value = np.random.randn(5, 1024).astype(np.float32)
        
        # Mock the LLM program summarization response
        async def mock_summarize(papers, keywords):
            return {
                "title": f"Program on {keywords[0]}",
                "summary": "This is a mock research program summary."
            }
        self.mock_llm.summarize_research_program.side_effect = mock_summarize
        
        dossier = {
            "name": "Jane Doe",
            "publications": [
                {"title": "Continual learning dynamics", "abstract": "continual learning framework"},
                {"title": "Curriculum learning theory", "abstract": "curriculum learning neural nets"},
                {"title": "Causal discovery representation", "abstract": "causal discovery representations"},
                {"title": "Representation similarity in RSA", "abstract": "rsa representational similarity analysis"},
                {"title": "Paper without abstract", "abstract": ""} # Noise paper
            ]
        }
        
        # Mock clustering to put first 2 papers in cluster 0, next 2 in cluster 1
        with patch.object(self.engine, "_cluster_papers", return_value=np.array([0, 0, 1, 1])):
            import asyncio
            programs = asyncio.run(self.engine._extract_programs_for_dossier(dossier, model_name="test-model"))
            
            # Expecting 3 programs: Cluster 0, Cluster 1, and the Noise paper program
            self.assertEqual(len(programs), 3)
            
            # Check program structure
            self.assertTrue(any("continual" in prog["title"].lower() or "learning" in prog["title"].lower() for prog in programs))
            self.assertTrue(any("causal" in prog["title"].lower() or "representation" in prog["title"].lower() for prog in programs))
            self.assertTrue(any(prog["title"] == "General & Exploratory Research" for prog in programs))

if __name__ == "__main__":
    unittest.main()
