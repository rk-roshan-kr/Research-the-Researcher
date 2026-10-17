import os
import shutil
import tempfile
import unittest
import numpy as np
from unittest.mock import MagicMock, patch

from academic_pipeline.embedding_cache import EmbeddingCache, cached_encode, get_embedding_cache

class TestEmbeddingCache(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_cache_save_and_retrieve(self):
        cache = EmbeddingCache(cache_dir=self.test_dir, db_name="test_emb.db")
        texts = ["hello world", "test embedding"]
        embeddings = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ], dtype=np.float32)
        
        # Save embeddings
        cache.save_embeddings(texts, embeddings, model_name="test-model", model_revision="main")
        
        # Retrieve embeddings
        retrieved = cache.get_embeddings(texts, model_name="test-model", model_revision="main")
        self.assertEqual(len(retrieved), 2)
        self.assertTrue(np.allclose(retrieved[0], [1.0, 2.0, 3.0]))
        self.assertTrue(np.allclose(retrieved[1], [4.0, 5.0, 6.0]))

    def test_cache_miss_and_invalidation(self):
        cache = EmbeddingCache(cache_dir=self.test_dir, db_name="test_emb.db")
        texts = ["hello world"]
        embeddings = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        
        cache.save_embeddings(texts, embeddings, model_name="test-model", model_revision="main")
        
        # Try to retrieve with different model name
        retrieved = cache.get_embeddings(texts, model_name="other-model", model_revision="main")
        self.assertEqual(retrieved, [None])
        
        # Try to retrieve with different model revision
        retrieved = cache.get_embeddings(texts, model_name="test-model", model_revision="v2")
        self.assertEqual(retrieved, [None])

    @patch("academic_pipeline.embedding_cache.get_embedding_cache")
    @patch("academic_pipeline.self_model.get_embedder_by_name")
    def test_cached_encode(self, mock_get_embedder, mock_get_cache):
        # Setup mock cache
        cache = EmbeddingCache(cache_dir=self.test_dir, db_name="test_emb.db")
        mock_get_cache.return_value = cache
        
        # Setup mock embedder
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        mock_get_embedder.return_value = mock_embedder
        
        texts = ["text1", "text2"]
        
        # First call: cache miss, calls encode
        embs1 = cached_encode(texts, model_name="mock-model", cache_dir=self.test_dir)
        self.assertEqual(embs1.shape, (2, 2))
        mock_embedder.encode.assert_called_once_with(
            texts, normalize_embeddings=True, batch_size=256, show_progress_bar=False
        )
        
        # Second call: cache hit, does not call encode again
        mock_embedder.reset_mock()
        embs2 = cached_encode(texts, model_name="mock-model", cache_dir=self.test_dir)
        self.assertEqual(embs2.shape, (2, 2))
        mock_embedder.encode.assert_not_called()
        self.assertTrue(np.allclose(embs1, embs2))

        # Partial cache call: text1 is cached, text3 is new
        mock_embedder.reset_mock()
        mock_embedder.encode.return_value = np.array([[0.5, 0.5]], dtype=np.float32)
        embs3 = cached_encode(["text1", "text3"], model_name="mock-model", cache_dir=self.test_dir)
        self.assertEqual(embs3.shape, (2, 2))
        mock_embedder.encode.assert_called_once_with(
            ["text3"], normalize_embeddings=True, batch_size=256, show_progress_bar=False
        )

if __name__ == "__main__":
    unittest.main()
