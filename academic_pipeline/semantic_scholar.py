import requests
import time
import logging

logger = logging.getLogger(__name__)

class SemanticScholarClient:
    """
    Client for interacting with the Semantic Scholar Academic Graph API.
    """
    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    def __init__(self, api_key=None):
        """
        Initialize the Semantic Scholar client.
        
        Args:
            api_key (str): Optional Semantic Scholar API key for higher rate limits.
        """
        self.api_key = api_key

    def _make_request(self, endpoint, params=None):
        """
        Helper method to make HTTP GET requests to the Semantic Scholar API.
        """
        if params is None:
            params = {}
            
        url = f"{self.BASE_URL}/{endpoint.lstrip('/')}"
        
        headers = {
            "User-Agent": "AcademicProfilePipeline/0.1"
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key
            
        max_retries = 5
        retry_delay = 2.0
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"Requesting Semantic Scholar API (Attempt {attempt+1}/{max_retries}): {url} with params {params}")
                response = requests.get(url, headers=headers, params=params, timeout=15)
                
                # Handle rate limiting (429) gracefully
                if response.status_code == 429:
                    if attempt == max_retries - 1:
                        response.raise_for_status()
                    retry_after = response.headers.get("Retry-After")
                    wait_time = int(retry_after) if retry_after and retry_after.isdigit() else (retry_delay * (2 ** attempt))
                    logger.warning(f"Semantic Scholar API rate limited (429). Waiting {wait_time}s before retrying...")
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
                    logger.error(f"Semantic Scholar request failed with status {status_code}: {e}")
                    raise
                
                if attempt == max_retries - 1:
                    logger.error(f"Semantic Scholar request failed after {max_retries} attempts: {e}")
                    raise
                
                import random
                wait_time = (retry_delay * (2 ** attempt)) + random.uniform(0.1, 1.0)
                logger.warning(f"Semantic Scholar request error: {e}. Retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)

    def search_authors(self, query):
        """
        Search for authors by name.
        
        Args:
            query (str): The author's name to search for.
            
        Returns:
            list: List of matching author summaries.
        """
        params = {
            "query": query,
            "fields": "name,externalIds,hIndex,paperCount,citationCount,affiliations"
        }
        try:
            data = self._make_request("author/search", params)
            return data.get("data", [])
        except Exception as e:
            logger.error(f"Error searching author on Semantic Scholar: {e}")
            return []

    def get_author_by_id(self, author_id):
        """
        Retrieve details for a specific author.
        
        Args:
            author_id (str): The Semantic Scholar Author ID.
            
        Returns:
            dict: The author profile details.
        """
        # Fetching basic info and external IDs
        fields = "name,affiliations,homepage,paperCount,citationCount,hIndex,externalIds"
        params = {
            "fields": fields
        }
        return self._make_request(f"author/{author_id}", params)

    def get_author_papers(self, author_id, max_papers=100):
        """
        Retrieve papers for an author using offset pagination.
        
        Args:
            author_id (str): Semantic Scholar Author ID.
            max_papers (int): Maximum number of papers to retrieve.
            
        Returns:
            list: List of papers with detailed metadata.
        """
        papers = []
        offset = 0
        limit = min(100, max_papers)
        
        fields = "title,abstract,citationCount,influentialCitationCount,year,venue,externalIds,authors,url,openAccessPdf"
        
        while len(papers) < max_papers:
            params = {
                "fields": fields,
                "offset": offset,
                "limit": limit
            }
            
            try:
                data = self._make_request(f"author/{author_id}/papers", params)
            except Exception as e:
                logger.error(f"Error fetching Semantic Scholar papers at offset {offset}: {e}")
                break
                
            results = data.get("data", [])
            if not results:
                break
                
            papers.extend(results)
            
            # If we got fewer results than the limit, we hit the end
            if len(results) < limit:
                break
                
            offset += len(results)
            
            # Rate limiting prevention for keyless access
            time.sleep(1.0 if not self.api_key else 0.1)
            
        return papers[:max_papers]
