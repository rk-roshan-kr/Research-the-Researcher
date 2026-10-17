import os
import shutil
import tempfile
import unittest
import time
from unittest.mock import patch, MagicMock

from academic_pipeline.cache_manager import CacheManager
from academic_pipeline.openalex import OpenAlexClient

class TestCrawledFaculty(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_save_and_retrieve_crawled_faculty(self):
        cache = CacheManager(cache_dir=self.test_dir)
        
        faculty_data = [
            {"id": "A1", "display_name": "Alice Smith", "summary_stats": {"h_index": 12}, "works_count": 45},
            {"id": "A2", "display_name": "Bob Jones", "h_index": 8, "works_count": 22},
        ]
        
        # Save faculty list
        cache.save_crawled_faculty("\u00c9cole Polytechnique F\u00e9d\u00e9rale de Lausanne", "EPFL", faculty_data)
        
        # Get count by exact name, alias, and case variations
        self.assertEqual(cache.get_crawled_faculty_count("\u00c9cole Polytechnique F\u00e9d\u00e9rale de Lausanne"), 2)
        self.assertEqual(cache.get_crawled_faculty_count("EPFL"), 2)
        self.assertEqual(cache.get_crawled_faculty_count("epfl"), 2)
        self.assertEqual(cache.get_crawled_faculty_count("\u00e9cole polytechnique f\u00e9d\u00e9rale de lausanne"), 2)
        self.assertEqual(cache.get_crawled_faculty_count("MIT"), 0)

        
        # List faculty profiles
        profiles = cache.list_crawled_faculty("EPFL")
        self.assertEqual(len(profiles), 2)
        
        # Check sorting by h_index (Alice has 12, Bob has 8)
        self.assertEqual(profiles[0]["name"], "Alice Smith")
        self.assertEqual(profiles[0]["h_index"], 12)
        self.assertEqual(profiles[1]["name"], "Bob Jones")
        self.assertEqual(profiles[1]["h_index"], 8)

    @patch("requests.get")
    def test_openalex_get_institution_authors(self, mock_get):
        # Setup mock response for paginated query
        mock_response_1 = MagicMock()
        mock_response_1.status_code = 200
        mock_response_1.json.return_value = {
            "results": [
                {"id": "A1", "display_name": "Alice Smith", "summary_stats": {"h_index": 12}, "works_count": 45},
            ],
            "meta": {
                "next_cursor": "cursor_token_1"
            }
        }
        
        mock_response_2 = MagicMock()
        mock_response_2.status_code = 200
        mock_response_2.json.return_value = {
            "results": [
                {"id": "A2", "display_name": "Bob Jones", "summary_stats": {"h_index": 8}, "works_count": 22},
            ],
            "meta": {
                "next_cursor": None
            }
        }
        
        mock_get.side_effect = [mock_response_1, mock_response_2]
        
        client = OpenAlexClient()
        authors = client.get_institution_authors("I5124864", min_h_index=5, max_authors=5)
        
        self.assertEqual(len(authors), 2)
        self.assertEqual(authors[0]["display_name"], "Alice Smith")
        self.assertEqual(authors[1]["display_name"], "Bob Jones")

if __name__ == "__main__":
    unittest.main()
