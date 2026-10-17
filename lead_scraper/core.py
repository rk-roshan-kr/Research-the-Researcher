import re
import urllib.parse
import logging
import asyncio
from typing import Set, List, Dict, Any, Optional, Tuple
import aiohttp
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Regular expressions for direct emails and obfuscated formats
EMAIL_REGEX = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b')

class WebEmailScraper:
    def __init__(self, concurrency_limit: int = 5, timeout: int = 15):
        self.semaphore = asyncio.Semaphore(concurrency_limit)
        self.timeout = timeout
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        }

    async def fetch_page(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        """Fetch the HTML source of a page with rate limiting."""
        async with self.semaphore:
            try:
                async with session.get(url, headers=self.headers, timeout=self.timeout) as response:
                    if response.status == 200:
                        return await response.text()
                    else:
                        logger.debug(f"Failed to fetch {url}: Status {response.status}")
            except Exception as e:
                logger.debug(f"Error fetching {url}: {e}")
        return None

    def determine_source_metadata(self, url: str) -> Tuple[str, str, int]:
        """
        Determine source type, contact permission, and confidence scoring based on URL patterns:
        - University directory: 100 confidence, faculty_public permission
        - Personal homepage: 95 confidence, personal_public permission
        - ORCID: 90 confidence, personal_public permission
        - Research lab page: 80 confidence, lab_public permission
        - General page: 60 confidence, unknown permission
        """
        parsed = urllib.parse.urlparse(url)
        path = parsed.path.lower()
        host = parsed.netloc.lower()

        if "orcid.org" in host:
            return "orcid", "personal_public", 90

        if any(keyword in path for keyword in ["faculty", "directory", "staff", "professor", "people", "contact-us"]):
            return "faculty_page", "faculty_public", 100

        if any(keyword in path for keyword in ["lab", "group", "research", "team"]):
            return "lab_page", "lab_public", 80

        # Heuristic for personal homepage
        if "~" in path or "github.io" in host or any(parsed.netloc.startswith(p) for p in ["www.", "personal."]) or len(parsed.netloc.split('.')) == 2:
            return "homepage", "personal_public", 95

        return "unknown", "unknown", 60

    def extract_emails_from_text(self, text: str) -> Set[str]:
        """Find direct emails in text."""
        found = set()
        for m in EMAIL_REGEX.findall(text):
            found.add(m.strip().lower())
        return found

    def decode_obfuscated_emails(self, text: str) -> Set[str]:
        """Attempts to find and reconstruct obfuscated emails."""
        found = set()
        obfuscated_regex = re.compile(
            r'\b([A-Za-z0-9._%+-]+)\s*[\(\[\\_]?\s*(?:at|AT|@)\s*[\)\]\\_]?\s*([A-Za-z0-9.-]+)\s*[\(\[\\_]?\s*(?:dot|DOT|\.)\s*[\)\]\\_]?\s*([A-Za-z]{2,7})\b'
        )
        for match in obfuscated_regex.finditer(text):
            parts = match.groups()
            email = f"{parts[0]}@{parts[1]}.{parts[2]}".replace(" ", "").lower()
            if EMAIL_REGEX.match(email):
                found.add(email)
        return found

    def extract_emails(self, html_content: str, url: str) -> List[Dict[str, Any]]:
        """Extract both direct and obfuscated emails and tag them with audit metadata."""
        emails_found = set()
        soup = BeautifulSoup(html_content, "html.parser")
        
        # 1. Search in mailto links
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if href.lower().startswith("mailto:"):
                email_part = href[7:].split("?")[0]
                email_part = urllib.parse.unquote(email_part)
                for m in EMAIL_REGEX.findall(email_part):
                    emails_found.add(m.strip().lower())

        # 2. Search in all page text (direct)
        page_text = soup.get_text(" ")
        emails_found.update(self.extract_emails_from_text(page_text))
        
        # 3. Search in all page text (obfuscated)
        emails_found.update(self.decode_obfuscated_emails(page_text))

        # Build records with metadata
        records = []
        source_type, permission, confidence = self.determine_source_metadata(url)
        for email in emails_found:
            records.append({
                "email": email,
                "email_source_url": url,
                "email_source_type": source_type,
                "confidence": confidence,
                "contact_permission": permission
            })
        return records

    async def scrape_url(self, url: str, deep_crawl: bool = True) -> List[Dict[str, Any]]:
        """
        Scrapes a URL for emails and returns list of dictionaries with metadata.
        If deep_crawl is True, visits related internal links.
        """
        records = []
        parsed_url = urllib.parse.urlparse(url)
        domain = parsed_url.netloc

        async with aiohttp.ClientSession() as session:
            html = await self.fetch_page(session, url)
            if not html:
                return records

            records.extend(self.extract_emails(html, url))

            if deep_crawl:
                soup = BeautifulSoup(html, "html.parser")
                candidate_links = set()
                
                for a in soup.find_all("a", href=True):
                    href = a["href"].strip()
                    full_link = urllib.parse.urljoin(url, href)
                    parsed_link = urllib.parse.urlparse(full_link)
                    
                    if parsed_link.netloc == domain:
                        link_path = parsed_link.path.lower()
                        if any(keyword in link_path for keyword in ["contact", "about", "people", "staff", "faculty", "team", "profile"]):
                            candidate_links.add(full_link)
                
                # Limit internal crawler to 4 subpages
                candidate_links = list(candidate_links)[:4]
                if candidate_links:
                    logger.debug(f"Deep crawling internal links for {url}: {candidate_links}")
                    tasks = [self.fetch_page(session, link) for link in candidate_links]
                    pages_content = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    for sub_url, sub_html in zip(candidate_links, pages_content):
                        if isinstance(sub_html, str) and sub_html:
                            records.extend(self.extract_emails(sub_html, sub_url))

        # Deduplicate list of dicts by email (keep highest confidence)
        dedup_records = {}
        for r in records:
            email = r["email"]
            if email not in dedup_records or r["confidence"] > dedup_records[email]["confidence"]:
                dedup_records[email] = r

        return list(dedup_records.values())
