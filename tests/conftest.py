import sys
import os
import types
import numpy as np

# Ensure the root package is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Predefined dictionary to cache word vectors for consistency
_WORD_VECTORS = {}

def get_word_vector(word, dim):
    if (word, dim) not in _WORD_VECTORS:
        import hashlib
        h_val = int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16) % (2**32 - 1)
        state = np.random.RandomState(h_val)
        vec = state.randn(dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        _WORD_VECTORS[(word, dim)] = vec / (norm if norm > 0 else 1.0)
    return _WORD_VECTORS[(word, dim)]


class MockSentenceTransformer:
    def __init__(self, model_name="BAAI/bge-large-en-v1.5", **kwargs):
        self.model_name = model_name
        self.device = kwargs.get("device", "cpu")

    def get_sentence_embedding_dimension(self):
        return 1024 if "bge-large" in self.model_name else 384

    def encode(self, texts, **kwargs):
        dim = self.get_sentence_embedding_dimension()
        res = []
        stopwords = {
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "by", "from", "is", "are", "was", "were",
            "be", "been", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "we", "our", "us",
            "it", "its", "their", "they", "as", "which", "such", "not",
            "also", "both", "can", "into", "than", "through", "using",
            "based", "show", "shows", "propose", "result", "results",
            "paper", "work", "method", "approach", "framework", "system",
            "model", "algorithm", "performance", "analysis", "evaluation",
            "experiments", "study"
        }
        for t in texts:
            import re
            words = re.findall(r'\b[a-z0-9]{3,}\b', str(t).lower())
            meaningful_words = [w for w in words if w not in stopwords]
            
            if not meaningful_words:
                meaningful_words = ["fallback"]

            sum_vec = np.zeros(dim, dtype=np.float32)
            for w in meaningful_words:
                sum_vec += get_word_vector(w, dim)

            norm_val = np.linalg.norm(sum_vec)
            sum_vec = sum_vec / (norm_val if norm_val > 0 else 1.0)
            res.append(sum_vec)
        return np.vstack(res)

class MockCrossEncoder:
    def __init__(self, model_name, **kwargs):
        self.model_name = model_name

    def predict(self, pairs, **kwargs):
        N = len(pairs)
        scores = np.zeros((N, 3), dtype=np.float32)
        for i, (p1, p2) in enumerate(pairs):
            p1_l = p1.lower()
            p2_l = p2.lower()
            # If it's a test checking for contradiction, return high contradiction score
            if any(w in p1_l or w in p2_l for w in ["not", "contradict", "conflict", "fail"]):
                scores[i] = [0.90, 0.05, 0.05]
            else:
                scores[i] = [0.05, 0.85, 0.10]
        return scores

# Register mock module in sys.modules
mock_st_module = types.ModuleType("sentence_transformers")
mock_st_module.SentenceTransformer = MockSentenceTransformer
mock_st_module.CrossEncoder = MockCrossEncoder
sys.modules["sentence_transformers"] = mock_st_module

# Redirect get_embedding_cache to a temporary directory in test mode to ensure database isolation
import tempfile
import academic_pipeline.embedding_cache

_TEST_CACHE_DIR = tempfile.mkdtemp()
original_get_embedding_cache = academic_pipeline.embedding_cache.get_embedding_cache

def mock_get_embedding_cache(cache_dir: str = "outputs"):
    if cache_dir == "outputs":
        return original_get_embedding_cache(_TEST_CACHE_DIR)
    return original_get_embedding_cache(cache_dir)

academic_pipeline.embedding_cache.get_embedding_cache = mock_get_embedding_cache

