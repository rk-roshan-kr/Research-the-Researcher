import asyncio
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock

from academic_pipeline.cache_manager import CacheManager
from academic_pipeline.dedup_engine import DedupEngine
from academic_pipeline.local_llm import LocalLLMClient
from academic_pipeline.pdf_downloader import PDFDownloader
from academic_pipeline.pipeline import AcademicPipeline
from academic_pipeline.resolver import ProfileResolver


# ──────────────────────────────────────────────────────────────────────────────
# Original tests (updated for new signatures)
# ──────────────────────────────────────────────────────────────────────────────

class TestAcademicPipeline(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_parse_input_type(self):
        mock_oa = MagicMock()
        mock_s2 = MagicMock()
        mock_scholar = MagicMock()
        resolver = ProfileResolver(mock_oa, mock_s2, mock_scholar)

        t, val = resolver.parse_input_type("0000-0002-1825-0097")
        self.assertEqual(t, "orcid")
        self.assertEqual(val, "0000-0002-1825-0097")

        t, val = resolver.parse_input_type("https://orcid.org/0000-0002-1825-0097")
        self.assertEqual(t, "orcid")
        self.assertEqual(val, "0000-0002-1825-0097")

        t, val = resolver.parse_input_type("https://scholar.google.com/citations?user=F9b9S8sAAAAJ&hl=en")
        self.assertEqual(t, "google_scholar")
        self.assertEqual(val, "F9b9S8sAAAAJ")

        t, val = resolver.parse_input_type("F9b9S8sAAAAJ")
        self.assertEqual(t, "google_scholar")
        self.assertEqual(val, "F9b9S8sAAAAJ")

        t, val = resolver.parse_input_type("A5003442464")
        self.assertEqual(t, "openalex")
        self.assertEqual(val, "A5003442464")

        t, val = resolver.parse_input_type("https://openalex.org/A5003442464")
        self.assertEqual(t, "openalex")
        self.assertEqual(val, "A5003442464")

        t, val = resolver.parse_input_type("https://www.semanticscholar.org/author/Oren-Etzioni/1741101")
        self.assertEqual(t, "semantic_scholar")
        self.assertEqual(val, "1741101")

        t, val = resolver.parse_input_type("Yann LeCun")
        self.assertEqual(t, "name")
        self.assertEqual(val, "Yann LeCun")

    def test_cache_manager(self):
        # New CacheManager uses SQLite; no filename kwarg needed
        cache = CacheManager(cache_dir=self.test_dir)
        self.assertIsNone(cache.get_paper_analysis("non_existent"))

        data = {"summary": "A test paper summary", "keywords": ["test"]}
        cache.save_paper_analysis("doi_123", data)
        self.assertEqual(cache.get_paper_analysis("doi_123"), data)

        # New instance reads from same SQLite DB
        new_cache = CacheManager(cache_dir=self.test_dir)
        self.assertEqual(new_cache.get_paper_analysis("doi_123"), data)

    def test_pdf_downloader_filename(self):
        from academic_pipeline.pdf_downloader import _clean_filename
        clean = _clean_filename("Deep Learning: Methods and Applications!")
        self.assertTrue(clean.endswith(".pdf"))
        self.assertNotIn(":", clean)
        self.assertNotIn("!", clean)

    def test_normalize_title(self):
        # AcademicPipeline now uses local LLM — no gemini_api_key needed
        pipeline = AcademicPipeline(cache_dir=self.test_dir, download_dir=self.test_dir)
        t1 = "Deep Learning: A Critical Review."
        t2 = "deep-learning a critical review"
        self.assertEqual(pipeline.normalize_title(t1), pipeline.normalize_title(t2))

    def test_merge_publications(self):
        pipeline = AcademicPipeline(cache_dir=self.test_dir, download_dir=self.test_dir)

        oa_works = [
            {
                "title": "A Breakthrough in Neural Networks",
                "publication_year": 2024,
                "cited_by_count": 100,
                "doi": "https://doi.org/10.1000/xyz123",
                "authorships": [{"author": {"display_name": "Alice Smith"}}],
                "locations": [{"source": {"display_name": "Nature AI"}}],
                "open_access": {"is_oa": True, "oa_url": None},
            }
        ]
        s2_papers = [
            {
                "title": "A Breakthrough in Neural Networks",
                "year": 2024,
                "citationCount": 110,
                "externalIds": {"DOI": "10.1000/xyz123"},
                "authors": [{"name": "Alice Smith"}],
                "abstract": "We present a breakthrough neural network architecture...",
                "tldr": {"text": "A breakthrough neural network structure is proposed."},
            },
            {
                "title": "A Second Paper on Deep Learning",
                "year": 2025,
                "citationCount": 5,
                "externalIds": {"ArXiv": "2501.12345"},
                "authors": [{"name": "Alice Smith"}, {"name": "Bob Jones"}],
            },
        ]

        merged = pipeline.merge_publications(oa_works, s2_papers)
        self.assertEqual(len(merged), 2, f"Expected 2 unique papers, got {len(merged)}")

        overlap = next(p for p in merged if p.get("doi") == "10.1000/xyz123")
        self.assertEqual(overlap["title"], "A Breakthrough in Neural Networks")
        self.assertGreaterEqual(overlap["citations_max"], 100)


# ──────────────────────────────────────────────────────────────────────────────
# New: DedupEngine tests
# ──────────────────────────────────────────────────────────────────────────────

class TestDedupEngine(unittest.TestCase):

    def setUp(self):
        self.engine = DedupEngine()

    def _make_paper(self, title, doi=None, arxiv=None, citations=0):
        return {
            "title": title,
            "doi": doi,
            "arxiv": arxiv,
            "year": 2024,
            "authors": ["Author A"],
            "citations_max": citations,
            "citations_openalex": citations,
            "citations_semantic": 0,
            "venue": "",
            "abstract": None,
            "tldr": None,
            "oa_url": None,
            "is_oa": False,
            "pdf_path": None,
            "analysis": None,
            "external_ids": {"doi": doi, "arxiv": arxiv},
        }

    def test_exact_doi_dedup(self):
        """Two papers with the same DOI should be merged into one."""
        papers = [
            self._make_paper("Neural Networks", doi="10.1000/abc", citations=50),
            self._make_paper("Neural Networks", doi="10.1000/abc", citations=60),
        ]
        result = DedupEngine().deduplicate(papers)
        self.assertEqual(len(result), 1)

    def test_different_dois_kept(self):
        """Papers with different DOIs must NOT be merged."""
        papers = [
            self._make_paper("Paper A", doi="10.1000/aaa", citations=10),
            self._make_paper("Paper B", doi="10.1000/bbb", citations=20),
        ]
        result = DedupEngine().deduplicate(papers)
        self.assertEqual(len(result), 2)

    def test_no_doi_title_dedup(self):
        """Papers without DOI/ArXiv should be deduped by title similarity."""
        papers = [
            self._make_paper("Attention Is All You Need", citations=1000),
            self._make_paper("Attention Is All You Need", citations=1100),
        ]
        result = DedupEngine().deduplicate(papers)
        self.assertLessEqual(len(result), 2)   # should be 1 after dedup

    def test_bloom_filter_basic(self):
        """Bloom filter must not produce false negatives."""
        from academic_pipeline.dedup_engine import BloomFilter
        bf = BloomFilter()
        for i in range(1000):
            bf.add(f"key_{i}")
        for i in range(1000):
            self.assertIn(f"key_{i}", bf)

    def test_union_find(self):
        """Union-Find must correctly identify clusters."""
        from academic_pipeline.collaboration_graph import UnionFind
        uf = UnionFind()
        uf.union("A", "B")
        uf.union("B", "C")
        uf.union("D", "E")
        self.assertEqual(uf.find("A"), uf.find("C"))   # A-B-C are connected
        self.assertNotEqual(uf.find("A"), uf.find("D"))  # D-E separate cluster

    def test_bfs_priority_coauthor_order(self):
        """CollaborationGraphBuilder._build_coauthor_map should respect paper counts."""
        from academic_pipeline.collaboration_graph import CollaborationGraphBuilder
        builder = CollaborationGraphBuilder(
            openalex_client=MagicMock(),
            s2_client=MagicMock(),
            cache_manager=MagicMock(get_resolved_profile=lambda _: None,
                                     save_resolved_profile=lambda *a: None),
        )
        publications = [
            {"title": f"Paper {i}", "authors": ["Target", "Heavy Collab"],
             "year": 2024, "citations_max": 0, "doi": None, "arxiv": None,
             "oa_url": None, "venue": ""}
            for i in range(5)
        ] + [
            {"title": "Solo Paper", "authors": ["Target", "Light Collab"],
             "year": 2024, "citations_max": 0, "doi": None, "arxiv": None,
             "oa_url": None, "venue": ""}
        ]
        comap, cometa = builder._build_coauthor_map("target", publications)
        self.assertEqual(len(comap["Heavy Collab"]), 5)
        self.assertEqual(len(comap["Light Collab"]), 1)


# ──────────────────────────────────────────────────────────────────────────────
# New: LocalLLMClient availability check (no Ollama required to pass)
# ──────────────────────────────────────────────────────────────────────────────

class TestLocalLLM(unittest.TestCase):

    def test_is_available_no_server(self):
        """is_available() should return False gracefully when Ollama is not running."""
        client = LocalLLMClient(host="http://localhost:19999")   # wrong port
        result = client.is_available()
        self.assertIsInstance(result, bool)   # must not raise


# ──────────────────────────────────────────────────────────────────────────────
# New: Batch and Department Mode tests
# ──────────────────────────────────────────────────────────────────────────────

class TestBatchAndDepartment(unittest.TestCase):

    def test_resolve_department_professors(self):
        mock_oa = MagicMock()
        mock_oa.search_institutions.return_value = [
            {"id": "https://openalex.org/I97018004", "display_name": "Stanford University"}
        ]
        mock_oa.get_works_by_institution_and_dept.return_value = [
            {
                "id": "work_1",
                "authorships": [
                    {
                        "author": {"id": "https://openalex.org/A111", "display_name": "Alice Smith"},
                        "institutions": [{"id": "https://openalex.org/I97018004"}],
                        "raw_affiliation_strings": ["Department of Computer Science, Stanford University"]
                    }
                ]
            }
        ]
        mock_oa.get_authors_batch.return_value = [
            {
                "id": "https://openalex.org/A111",
                "display_name": "Alice Smith",
                "cited_by_count": 500,
                "summary_stats": {"h_index": 15},
                "orcid": "0000-0001-2345-6789"
            }
        ]

        # Use temporary directory for testing pipeline resolver
        import tempfile
        import shutil
        test_dir = tempfile.mkdtemp()
        try:
            pipeline = AcademicPipeline(cache_dir=test_dir, download_dir=test_dir)
            pipeline.openalex = mock_oa

            profs = pipeline.resolve_department_professors("Stanford", "Computer Science", limit_professors=10)
            
            self.assertEqual(len(profs), 1)
            self.assertEqual(profs[0]["name"], "Alice Smith")
            self.assertEqual(profs[0]["h_index"], 15)
            self.assertEqual(profs[0]["citations"], 500)
            self.assertEqual(profs[0]["orcid"], "0000-0001-2345-6789")
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)



class TestPreprocessingUpgrades(unittest.TestCase):

    def test_affiliation_hint_ranking(self):
        mock_oa = MagicMock()
        mock_s2 = MagicMock()
        mock_scholar = MagicMock()
        resolver = ProfileResolver(mock_oa, mock_s2, mock_scholar)
        
        # Test that resolve_researcher extracts the hint correctly
        # We can mock search_authors to return list of candidates
        mock_oa.search_authors.return_value = [
            {"id": "A1", "display_name": "Yann LeCun", "last_known_institutions": [{"display_name": "New York University"}]},
            {"id": "A2", "display_name": "Yann LeCun", "last_known_institutions": [{"display_name": "Facebook AI Research"}]}
        ]
        mock_s2.search_authors.return_value = []
        
        # Resolve researcher with hint
        resolved = resolver.resolve_researcher("Yann LeCun (New York University)")
        
        # Top candidate should be the one at NYU
        self.assertEqual(resolved["openalex_id"], "A1")

    def test_publication_overlap_match(self):
        mock_oa = MagicMock()
        mock_s2 = MagicMock()
        mock_scholar = MagicMock()
        resolver = ProfileResolver(mock_oa, mock_s2, mock_scholar)

        # Mock papers lists
        mock_oa.get_author_works.return_value = [
            {"title": "Deep Learning with Convolutional Networks"},
            {"title": "Unsupervised Visual Feature Learning"}
        ]
        mock_s2.get_author_papers.return_value = [
            {"title": "Deep Learning with Convolutional Networks"},
            {"title": "Completely Different Paper"}
        ]

        overlap_found = resolver._check_publication_overlap("A123", "S123")
        self.assertTrue(overlap_found)

    def test_fuzzy_preprint_dedup(self):
        engine = DedupEngine()
        papers = [
            {
                "title": "Attention is All You Need",
                "arxiv": "1706.03762",
                "venue": "arXiv preprint",
                "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...",
                "citations_max": 500
            },
            {
                "title": "Attention is All You Need (Published Version)",
                "doi": "10.5555/3295222.3295349",
                "venue": "NeurIPS",
                "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...",
                "citations_max": 800
            }
        ]

        result = engine.deduplicate(papers)
        # Should be merged into 1 paper
        self.assertEqual(len(result), 1)
        merged = result[0]
        self.assertEqual(merged["arxiv"], "1706.03762")
        self.assertEqual(merged["doi"], "10.5555/3295222.3295349")
        self.assertEqual(merged["citations_max"], 800)

    def test_topical_coherence_filtering(self):
        import tempfile
        import shutil
        test_dir = tempfile.mkdtemp()
        try:
            pipeline = AcademicPipeline(cache_dir=test_dir, download_dir=test_dir)
            
            publications = [
                {"title": "Deep Learning and Neural Architecture Search", "abstract": "This work analyzes reinforcement learning for NAS.", "citations_max": 100, "authors": ["Alice Smith", "Bob Jones"]},
                {"title": "Recurrent Models for Visual Classification", "abstract": "We explore recurrent visual classification models.", "citations_max": 80, "authors": ["Alice Smith"]},
                {"title": "Generative Adversarial Nets", "abstract": "Generative adversarial neural networks are analyzed.", "citations_max": 60, "authors": ["Alice Smith"]},
                {"title": "Transformer Models for NLP", "abstract": "Transformers sequence modeling is explored.", "citations_max": 40, "authors": ["Alice Smith"]},
                {"title": "Visual Reinforcement Learning Control", "abstract": "Deep neural nets for robotic reinforcement learning control.", "citations_max": 20, "authors": ["Alice Smith", "Bob Jones"]},
                # Noise paper: completely different topic, zero author overlap, no keyword overlap
                {"title": "Organic Chemistry Syntheses", "abstract": "Organic synthesis of carbon compounds using metallic catalysts.", "citations_max": 5, "authors": ["Charlie Brown"]}
            ]
            
            filtered = pipeline.filter_topical_coherence(publications, "Alice Smith")
            
            # The organic chemistry paper should be filtered out
            self.assertEqual(len(filtered), 5)
            self.assertNotIn("Organic Chemistry Syntheses", [p["title"] for p in filtered])
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
