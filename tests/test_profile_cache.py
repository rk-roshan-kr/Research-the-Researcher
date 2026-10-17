import os
import shutil
import tempfile
import unittest
import time
from unittest.mock import patch

from academic_pipeline.profile_cache import ProfileCache

class TestProfileCache(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_save_and_retrieve_profile(self):
        cache = ProfileCache(cache_dir=self.test_dir, db_name="test_profiles.db", ttl_days=1.0)
        
        prof_id = "A123"
        dossier = {
            "name": "Jane Doe",
            "openalex_id": "A123",
            "position": "Associate Professor",
            "publications": [{"title": "Paper 1", "year": 2023}]
        }
        programs = [
            {
                "title": "Machine Learning",
                "summary": "Focus on neural networks",
                "publications": [{"title": "Paper 1", "year": 2023}]
            }
        ]

        # Save profile
        cache.save_profile(prof_id, dossier, programs)

        # Retrieve profile
        retrieved = cache.get_profile(prof_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["dossier"]["name"], "Jane Doe")
        self.assertEqual(len(retrieved["programs"]), 1)
        self.assertEqual(retrieved["programs"][0]["title"], "Machine Learning")

    def test_profile_ttl_expiration(self):
        # 1 day TTL
        cache = ProfileCache(cache_dir=self.test_dir, db_name="test_profiles.db", ttl_days=1.0)
        
        prof_id = "A123"
        dossier = {"name": "Jane Doe"}
        programs = []

        cache.save_profile(prof_id, dossier, programs)

        # Retrieve immediately -> success
        self.assertIsNotNone(cache.get_profile(prof_id))

        # Retrieve 1.5 days later -> expired (returns None)
        future_time = time.time() + 1.5 * 86400
        with patch("time.time", return_value=future_time):
            self.assertIsNone(cache.get_profile(prof_id))

    def test_corrupted_json_handling(self):
        cache = ProfileCache(cache_dir=self.test_dir, db_name="test_profiles.db")
        prof_id = "A999"
        
        # Manually insert corrupted JSON string
        import sqlite3
        conn = sqlite3.connect(cache.db_path)
        conn.execute(
            "INSERT INTO professor_profiles (professor_id, dossier, programs, last_refresh) VALUES (?, ?, ?, ?)",
            (prof_id, "{invalid json", "[]", time.time())
        )
        conn.commit()
        conn.close()

        # Should return None gracefully without raising json decode error
        retrieved = cache.get_profile(prof_id)
        self.assertIsNone(retrieved)

if __name__ == "__main__":
    unittest.main()
