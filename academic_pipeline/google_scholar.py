import requests
from bs4 import BeautifulSoup
import logging
import re

logger = logging.getLogger(__name__)

class GoogleScholarScraper:
    """
    Scrapes researcher metrics and publications from public Google Scholar profile pages.
    Treats scraping as a fallback and fails gracefully if blocked by Google.
    """
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9"
        }

    def _extract_id_from_url(self, url_or_id):
        """Extract Google Scholar User ID from a URL or return the string if it matches ID format."""
        if not url_or_id:
            return None
        
        # User ID is typically 12 characters alphanumeric/dashes
        if len(url_or_id) == 12 and re.match(r'^[a-zA-Z0-9_\-]+$', url_or_id):
            return url_or_id
            
        # Parse from URL
        match = re.search(r'[?&]user=([^&]+)', url_or_id)
        if match:
            return match.group(1)
            
        return None

    def scrape_profile(self, url_or_id):
        """
        Scrapes a researcher's Google Scholar profile page.
        
        Args:
            url_or_id (str): Google Scholar profile URL or User ID.
            
        Returns:
            dict: Scraped metrics and metadata, or None if blocked/failed.
        """
        user_id = self._extract_id_from_url(url_or_id)
        if not user_id:
            logger.error(f"Could not extract Google Scholar User ID from: {url_or_id}")
            return None
            
        url = f"https://scholar.google.com/citations?user={user_id}&hl=en&pagesize=100"
        
        try:
            logger.info(f"Attempting to scrape Google Scholar profile: {url}")
            response = requests.get(url, headers=self.headers, timeout=15)
            
            if response.status_code == 429:
                logger.warning("Google Scholar scraping blocked by rate limits (429).")
                return None
                
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Check if page is valid profile page
            name_node = soup.select_one("#gsc_prf_in")
            if not name_node:
                logger.warning("Google Scholar page structure not recognized or CAPTCHA triggered.")
                return None
                
            name = name_node.text.strip()
            
            # Extract affiliation
            affiliation_node = soup.select_one(".gsc_prf_il")
            affiliation = affiliation_node.text.strip() if affiliation_node else ""
            
            # Extract image
            img_node = soup.select_one("#gsc_prf_pup-art img")
            photo_url = img_node['src'] if img_node and img_node.get('src') else None
            if photo_url and photo_url.startswith("/"):
                photo_url = "https://scholar.google.com" + photo_url
                
            # Extract interests
            interests = []
            for a in soup.select("#gsc_prf_int a"):
                interests.append(a.text.strip())
                
            # Extract citation metrics
            metrics = {
                "citations": {"all": 0, "since_2021": 0},
                "h_index": {"all": 0, "since_2021": 0},
                "i10_index": {"all": 0, "since_2021": 0}
            }
            
            metric_rows = soup.select("#gsc_rsb_st tbody tr")
            if len(metric_rows) >= 3:
                # Row 0: Citations
                cols_citations = metric_rows[0].select("td")
                if len(cols_citations) >= 3:
                    metrics["citations"]["all"] = int(cols_citations[1].text.strip() or 0)
                    metrics["citations"]["since_2021"] = int(cols_citations[2].text.strip() or 0)
                    
                # Row 1: h-index
                cols_h = metric_rows[1].select("td")
                if len(cols_h) >= 3:
                    metrics["h_index"]["all"] = int(cols_h[1].text.strip() or 0)
                    metrics["h_index"]["since_2021"] = int(cols_h[2].text.strip() or 0)
                    
                # Row 2: i10-index
                cols_i10 = metric_rows[2].select("td")
                if len(cols_i10) >= 3:
                    metrics["i10_index"]["all"] = int(cols_i10[1].text.strip() or 0)
                    metrics["i10_index"]["since_2021"] = int(cols_i10[2].text.strip() or 0)
            
            # Extract top papers (visible on page)
            papers = []
            for tr in soup.select(".gsc_a_tr"):
                title_node = tr.select_one(".gsc_a_at")
                if not title_node:
                    continue
                    
                title = title_node.text.strip()
                link = "https://scholar.google.com" + title_node['href'] if title_node.get('href') else None
                
                # Gray text fields (authors and venue)
                gray_fields = tr.select(".gsc_a_t .gs_gray")
                authors = gray_fields[0].text.strip() if len(gray_fields) > 0 else ""
                venue = gray_fields[1].text.strip() if len(gray_fields) > 1 else ""
                
                # Citations
                citations_node = tr.select_one(".gsc_a_ac")
                citations = 0
                if citations_node and citations_node.text.strip():
                    try:
                        citations = int(citations_node.text.strip())
                    except ValueError:
                        pass
                        
                # Year
                year_node = tr.select_one(".gsc_a_hc")
                year = None
                if year_node and year_node.text.strip():
                    try:
                        year = int(year_node.text.strip())
                    except ValueError:
                        pass
                        
                papers.append({
                    "title": title,
                    "url": link,
                    "authors": authors,
                    "venue": venue,
                    "citations": citations,
                    "year": year
                })
                
            return {
                "scholar_id": user_id,
                "name": name,
                "affiliation": affiliation,
                "photo_url": photo_url,
                "interests": interests,
                "metrics": metrics,
                "papers": papers
            }
            
        except Exception as e:
            logger.error(f"Error scraping Google Scholar: {e}")
            return None
