import requests
import time
import logging
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)

class OpenAlexClient:
    """
    Client for interacting with the OpenAlex API (https://api.openalex.org).
    """
    BASE_URL = "https://api.openalex.org"

    def __init__(self, api_key=None, email=None):
        """
        Initialize the OpenAlex client.
        
        Args:
            api_key (str): Optional API key (recommended after Feb 2026).
            email (str): Optional email for the polite pool (optional fallback).
        """
        self.api_key = api_key
        self.email = email

    def _make_request(self, endpoint, params=None):
        """
        Helper method to make HTTP GET requests to the OpenAlex API.
        """
        if params is None:
            params = {}
        
        # Inject API key if configured
        if self.api_key:
            params["api_key"] = self.api_key
        elif self.email:
            # Polite pool parameter
            params["mailto"] = self.email
            
        url = f"{self.BASE_URL}/{endpoint.lstrip('/')}"
        
        headers = {
            "User-Agent": "AcademicProfilePipeline/0.1 (mailto:researcher-pipeline@example.com)"
        }
        if self.email:
            headers["User-Agent"] = f"AcademicProfilePipeline/0.1 (mailto:{self.email})"
            
        max_retries = 5
        retry_delay = 2.0
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"Requesting OpenAlex API (Attempt {attempt+1}/{max_retries}): {url} with params {params}")
                response = requests.get(url, headers=headers, params=params, timeout=15)
                
                # Handle rate limiting (429) gracefully
                if response.status_code == 429:
                    if attempt == max_retries - 1:
                        response.raise_for_status()
                    wait_time = retry_delay * (2 ** attempt)
                    retry_after = response.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        wait_time = max(wait_time, int(retry_after))
                    logger.warning(f"OpenAlex API rate limited (429). Retrying in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                status_code = None
                if hasattr(e, 'response') and e.response is not None:
                    status_code = e.response.status_code
                
                # If it's a client error other than 429, fail immediately
                if status_code and status_code not in (429, 500, 502, 503, 504):
                    logger.error(f"OpenAlex request failed with status {status_code}: {e}")
                    raise
                
                if attempt == max_retries - 1:
                    logger.error(f"OpenAlex request failed after {max_retries} attempts: {e}")
                    raise
                
                import random
                wait_time = (retry_delay * (2 ** attempt)) + random.uniform(0.1, 1.0)
                logger.warning(f"OpenAlex request error: {e}. Retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)

    def search_authors(self, query):
        """
        Search for authors by name.
        
        Args:
            query (str): The search query (author name).
            
        Returns:
            list: List of matching author objects.
        """
        # Using search parameter as recommended in modern OpenAlex API docs
        data = self._make_request("authors", {"search": query})
        return data.get("results", [])

    def get_author_by_id(self, author_id):
        """
        Retrieve a specific author profile by OpenAlex ID.
        
        Args:
            author_id (str): OpenAlex ID (e.g. 'A5003442464' or full URL).
            
        Returns:
            dict: The author profile object.
        """
        # Extract ID suffix if full URL is passed
        clean_id = author_id.split("/")[-1]
        return self._make_request(f"authors/{clean_id}")

    def get_author_by_orcid(self, orcid):
        """
        Retrieve a specific author profile by ORCID.
        
        Args:
            orcid (str): The researcher's ORCID iD (e.g. '0000-0002-1825-0097' or full URL).
            
        Returns:
            dict: The author profile object or None if not found.
        """
        # Clean ORCID to format 0000-0000-0000-0000
        clean_orcid = orcid.split("/")[-1]
        
        # Check if it has a prefix like orcid:
        if clean_orcid.lower().startswith("orcid:"):
            clean_orcid = clean_orcid[6:]
            
        # We can fetch directly using the orcid: prefix endpoint
        try:
            return self._make_request(f"authors/orcid:{clean_orcid}")
        except Exception as e:
            logger.warning(f"Failed direct ORCID lookup for {clean_orcid}: {e}. Trying filter...")
            
        # Fallback to search filter
        try:
            data = self._make_request("authors", {"filter": f"orcid:{clean_orcid}"})
            results = data.get("results", [])
            if results:
                return results[0]
        except Exception as e:
            logger.error(f"Failed to find author by ORCID filter: {e}")
            
        return None

    def get_author_works(self, author_id, max_works=100):
        """
        Retrieve all works associated with an author using cursor-based pagination.
        
        Args:
            author_id (str): OpenAlex ID.
            max_works (int): Limit on the number of works to retrieve.
            
        Returns:
            list: List of work objects.
        """
        clean_id = author_id.split("/")[-1]
        
        works = []
        cursor = "*"
        per_page = min(100, max_works)
        
        while len(works) < max_works:
            params = {
                "filter": f"author.id:{clean_id}",
                "per_page": per_page,
                "cursor": cursor
            }
            
            try:
                data = self._make_request("works", params)
            except Exception as e:
                logger.error(f"Error fetching works page: {e}")
                break
                
            results = data.get("results", [])
            if not results:
                break
                
            works.extend(results)
            
            # Check for next cursor
            next_cursor = data.get("meta", {}).get("next_cursor")
            if not next_cursor or next_cursor == cursor:
                break
                
            cursor = next_cursor
            
            # Simple sleep to prevent hammering the API
            time.sleep(0.1)
            
        return works[:max_works]

    def get_authors_batch(self, author_ids: list[str]) -> list[dict]:
        """
        Retrieve multiple author profiles by OpenAlex ID in a single query.
        
        Args:
            author_ids (list): List of OpenAlex IDs.
            
        Returns:
            list: List of author profile objects.
        """
        if not author_ids:
            return []
        
        # Clean and deduplicate IDs
        clean_ids = list(set(aid.split("/")[-1] for aid in author_ids if aid))
        
        results = []
        # Chunk requests because OpenAlex filter query has length limits (typically 50 IDs)
        for i in range(0, len(clean_ids), 50):
            chunk = clean_ids[i:i+50]
            filter_str = "|".join(chunk)
            try:
                data = self._make_request("authors", {"filter": f"openalex_id:{filter_str}", "per_page": 50})
                results.extend(data.get("results", []))
            except Exception as e:
                logger.error(f"Error in OpenAlex batch authors fetch: {e}")
                
        return results

    def search_institutions(self, query):
        """
        Search for institutions by name.
        """
        data = self._make_request("institutions", {"search": query})
        return data.get("results", [])

    def get_works_by_institution_and_dept(self, inst_id, dept_name, max_works=200):
        """
        Retrieve works for a specific institution filtered by department name.
        """
        clean_inst_id = inst_id.split("/")[-1]
        
        works = []
        cursor = "*"
        per_page = min(100, max_works)
        
        # We try raw_affiliation_strings.search first, fallback to raw_affiliation_string.search if it fails
        filter_keys = ["raw_affiliation_strings.search", "raw_affiliation_string.search"]
        
        for filter_key in filter_keys:
            works = []
            cursor = "*"
            failed = False
            while len(works) < max_works:
                params = {
                    "filter": f"institutions.id:{clean_inst_id},{filter_key}:{dept_name}",
                    "per_page": per_page,
                    "cursor": cursor
                }
                try:
                    data = self._make_request("works", params)
                except Exception as e:
                    logger.warning(f"Error fetching works with filter {filter_key}: {e}. Trying next option...")
                    failed = True
                    break
                    
                results = data.get("results", [])
                if not results:
                    break
                    
                works.extend(results)
                
                next_cursor = data.get("meta", {}).get("next_cursor")
                if not next_cursor or next_cursor == cursor:
                    break
                cursor = next_cursor
                time.sleep(0.1)
                
            if not failed and works:
                # Successfully retrieved works using this filter, return them
                return works[:max_works]
                
        return works[:max_works]

    def get_authors_by_institution_and_concepts(self, inst_id: str, concept_ids: list[str], max_authors: int = 100) -> list[dict]:
        """
        Retrieve authors associated with an institution, filtered by relevant concept IDs by querying works.
        """
        clean_inst_id = inst_id.split("/")[-1]
        
        candidate_ids = set()
        author_frequencies = {}
        
        for concept in concept_ids:
            for sort_order in ["cited_by_count:desc", "publication_date:desc"]:
                params = {
                    "filter": f"institutions.id:{clean_inst_id},concepts.id:{concept}",
                    "sort": sort_order,
                    "per_page": 40
                }
                try:
                    data = self._make_request("works", params)
                    results = data.get("results", [])
                    for w in results:
                        for auth in w.get("authorships", []):
                            inst_ids = [inst.get("id") for inst in auth.get("institutions", [])]
                            is_affiliated = any(clean_inst_id in iid for iid in inst_ids if iid)
                            if is_affiliated:
                                author_id = auth.get("author", {}).get("id")
                                if author_id:
                                    candidate_ids.add(author_id)
                                    author_frequencies[author_id] = author_frequencies.get(author_id, 0) + 1
                except Exception as e:
                    logger.error(f"Error fetching works for author discovery concept {concept}: {e}")
                
        author_profiles = self.get_authors_batch(list(candidate_ids))
        
        filtered_profiles = []
        for profile in author_profiles:
            summary_stats = profile.get("summary_stats", {})
            h_index = profile.get("h_index") or summary_stats.get("h_index") or 0
            works_count = profile.get("works_count") or 0
            if h_index >= 5 and works_count >= 10:
                filtered_profiles.append(profile)
                
        # Sort by frequency of matching works descending, tie-break by total works count descending
        filtered_profiles.sort(key=lambda x: (author_frequencies.get(x.get("id"), 0), x.get("works_count", 0)), reverse=True)
        
        return filtered_profiles[:max_authors]

    def get_institution_authors(self, inst_id: str, min_h_index: int = 5, max_authors: int = 5000) -> list[dict]:
        """
        Retrieve all authors associated with an institution using cursor-based pagination.
        
        Args:
            inst_id (str): OpenAlex ID.
            min_h_index (int): Minimum h_index to filter faculty.
            max_authors (int): Cap on the number of authors to retrieve.
            
        Returns:
            list: List of author objects.
        """
        clean_id = inst_id.split("/")[-1]
        
        authors = []
        cursor = "*"
        per_page = 100
        
        filter_str = f"last_known_institutions.id:{clean_id}"
        if min_h_index > 0:
            filter_str += f",summary_stats.h_index:>{min_h_index}"
            
        while len(authors) < max_authors:
            params = {
                "filter": filter_str,
                "per_page": per_page,
                "cursor": cursor
            }
            
            try:
                data = self._make_request("authors", params)
            except Exception as e:
                logger.error(f"Error fetching authors page: {e}")
                break
                
            results = data.get("results", [])
            if not results:
                break
                
            authors.extend(results)
            
            # Check for next cursor
            next_cursor = data.get("meta", {}).get("next_cursor")
            if not next_cursor or next_cursor == cursor:
                break
                
            cursor = next_cursor
            
            # Simple sleep to prevent hammering the API
            time.sleep(0.1)
            
        return authors[:max_authors]

