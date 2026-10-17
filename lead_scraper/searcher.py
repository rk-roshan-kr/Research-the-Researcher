import logging
from typing import List, Dict, Any
from academic_pipeline.local_search import LocalSearchClient

logger = logging.getLogger(__name__)

class LeadSearcher:
    def __init__(self, searxng_url: str = "http://localhost:8080"):
        self.search_client = LocalSearchClient(searxng_url=searxng_url)

    def search_university_directories(self, university: str, department: str = "computer science") -> List[Dict[str, Any]]:
        """
        Search for faculty directory pages for a given university.
        E.g. site:mit.edu "faculty directory" OR "staff list" OR "people" "computer science"
        """
        domain_map = {
            "mit": "mit.edu",
            "stanford": "stanford.edu",
            "cmu": "cs.cmu.edu",
            "berkeley": "berkeley.edu",
            "eth zurich": "ethz.ch",
            "epfl": "epfl.ch",
            "oxford": "ox.ac.uk",
            "cambridge": "cam.ac.uk",
            "iit": "iitb.ac.in",  # generic, or we search by name
        }
        
        domain = domain_map.get(university.lower().strip())
        site_filter = f"site:{domain}" if domain else f'"{university}"'
        
        query = f'{site_filter} "faculty directory" OR "people" OR "staff" "{department}"'
        logger.info(f"Searching university directories: '{query}'")
        return self.search_client.search(query, num_results=8)

    def search_conference_authors(self, conference: str, topic: str) -> List[Dict[str, Any]]:
        """
        Search for author profiles or accepted paper pages at a conference.
        E.g. "NeurIPS" "continual learning" site:nips.cc OR site:openreview.net
        """
        conf_domains = {
            "neurips": "site:neurips.cc OR site:openreview.net",
            "icml": "site:icml.cc OR site:openreview.net",
            "iclr": "site:openreview.net",
            "cvpr": "site:thecvf.com",
            "acl": "site:aclanthology.org"
        }
        
        domain_filter = conf_domains.get(conference.lower().strip(), f'"{conference}"')
        query = f'{domain_filter} "{topic}" paper OR authors'
        logger.info(f"Searching conference papers: '{query}'")
        return self.search_client.search(query, num_results=8)

    def search_company_pages(self, company: str) -> List[Dict[str, Any]]:
        """
        Search for team/engineering staff pages for key AI companies.
        E.g. site:openai.com "team" OR "people" OR "researchers"
        """
        company_domains = {
            "openai": "openai.com",
            "anthropic": "anthropic.com",
            "google deepmind": "deepmind.google",
            "microsoft research": "microsoft.com/en-us/research",
            "meta fair": "fair.meta.com",
            "nvidia research": "nvidia.com/en-us/research"
        }
        
        domain = company_domains.get(company.lower().strip())
        site_filter = f"site:{domain}" if domain else f'"{company}"'
        query = f'{site_filter} "team" OR "people" OR "staff" OR "researchers"'
        logger.info(f"Searching company team pages: '{query}'")
        return self.search_client.search(query, num_results=8)

    def search_creators(self, niche: str = "AI newsletter") -> List[Dict[str, Any]]:
        """Search for newsletters, bloggers, and podcasters in a niche."""
        query = f'"{niche}" OR "AI blogger" OR "AI podcast" site:substack.com OR site:medium.com OR site:youtube.com'
        logger.info(f"Searching AI newsletter creators: '{query}'")
        return self.search_client.search(query, num_results=10)
