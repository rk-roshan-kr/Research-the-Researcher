import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from academic_pipeline.openalex import OpenAlexClient

logger = logging.getLogger(__name__)

class LeadOpenAlexSearcher:
    def __init__(self, email: Optional[str] = "researcher-pipeline@example.com"):
        self.client = OpenAlexClient(email=email)

    def search_leads_by_concept(self, concept_query: str, limit: int = 20, use_fallback: bool = True) -> List[Dict[str, Any]]:
        """
        Search OpenAlex works matching a concept/topic query,
        extract author profiles in parallel, fetch telemetry,
        and verify official contact details.
        """
        logger.info(f"Querying OpenAlex works for topic: '{concept_query}' with limit {limit} (parallelized)")
        
        leads = []
        seen_author_ids_processed = set()
        
        cursor = "*"
        per_page = 100
        
        while len(leads) < limit:
            params = {
                "filter": f"title.search:{concept_query}",
                "sort": "cited_by_count:desc",
                "per_page": per_page,
                "cursor": cursor
            }
            
            try:
                data = self.client._make_request("works", params)
                works = data.get("results", [])
                if not works:
                    break
            except Exception as e:
                logger.error(f"Error querying OpenAlex works: {e}")
                break

            # 1. Gather all unique candidate author IDs on this works page
            page_author_ids = []
            for work in works:
                for auth in work.get("authorships", []):
                    author_id = auth.get("author", {}).get("id", "")
                    if author_id and author_id not in seen_author_ids_processed:
                        page_author_ids.append(author_id)
            
            # Deduplicate the IDs for parallel fetch
            page_author_ids = list(set(page_author_ids))
            
            # 2. Fetch profiles in parallel
            profiles_map = {}
            if page_author_ids:
                logger.info(f"Fetching {len(page_author_ids)} author profiles in parallel using ThreadPoolExecutor...")
                with ThreadPoolExecutor(max_workers=20) as executor:
                    future_to_id = {executor.submit(self.enrich_author_profile, aid): aid for aid in page_author_ids}
                    for future in as_completed(future_to_id):
                        aid = future_to_id[future]
                        try:
                            profiles_map[aid] = future.result()
                        except Exception as e:
                            logger.error(f"Error fetching profile for {aid}: {e}")
                            profiles_map[aid] = {}

            # 3. Process works and match details
            for work in works:
                recent_paper_title = work.get("title", "")
                pub_date = work.get("publication_date", "")
                corresponding_emails = work.get("corresponding_author_emails", [])
                
                authorships = work.get("authorships", [])
                for auth in authorships:
                    author_info = auth.get("author", {})
                    author_id = author_info.get("id", "")
                    
                    if not author_id or author_id in seen_author_ids_processed:
                        continue
                    
                    seen_author_ids_processed.add(author_id)
                    
                    profile_details = profiles_map.get(author_id, {})
                    if not profile_details:
                        continue
                    
                    author_name = author_info.get("display_name", "")
                    email_found = None
                    source_url = f"https://openalex.org/{author_id.split('/')[-1]}"
                    source_type = "unknown"
                    confidence = 0
                    permission = "unknown"
                    
                    if corresponding_emails:
                        name_tokens = author_name.lower().split()
                        for email in corresponding_emails:
                            email_prefix = email.split("@")[0].lower()
                            if any(token in email_prefix for token in name_tokens):
                                email_found = email
                                source_url = work.get("id", source_url)
                                source_type = "orcid" if "orcid" in source_url else "faculty_page"
                                confidence = 90
                                permission = "personal_public"
                                break

                    institutions = auth.get("institutions", [])
                    inst_name = institutions[0].get("display_name", "") if institutions else ""
                    if not inst_name:
                        inst_name = auth.get("raw_affiliation_string", "")

                    orcid = author_info.get("orcid", "")
                    if orcid and not email_found:
                        source_url = orcid
                        source_type = "orcid"
                        permission = "personal_public"
                        confidence = 90

                    if not email_found and use_fallback:
                        from lead_scraper.searcher import LeadSearcher
                        from lead_scraper.core import WebEmailScraper
                        import asyncio
                        
                        searcher = LeadSearcher()
                        scraper = WebEmailScraper()
                        search_query = f'"{author_name}" {inst_name} email'
                        logger.info(f"Email fallback: searching web for {author_name} contact info...")
                        try:
                            results = searcher.search_client.search(search_query, num_results=3)
                            if not results:
                                search_query_simple = f'"{author_name}" email'
                                logger.info(f"Email fallback: retrying with simpler query '{search_query_simple}'")
                                results = searcher.search_client.search(search_query_simple, num_results=3)
                            try:
                                loop = asyncio.get_event_loop()
                            except RuntimeError:
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                                
                            for r in results:
                                target_url = r.get("url", "")
                                if not target_url:
                                    continue
                                scraped = loop.run_until_complete(scraper.scrape_url(target_url, deep_crawl=True))
                                
                                name_tokens_fl = [t for t in author_name.lower().split() if len(t) > 2]
                                for rec in scraped:
                                    email_prefix = rec["email"].split("@")[0].lower()
                                    if any(t in email_prefix for t in name_tokens_fl):
                                        email_found = rec["email"]
                                        source_url = rec["email_source_url"]
                                        source_type = rec["email_source_type"]
                                        confidence = rec["confidence"]
                                        permission = rec["contact_permission"]
                                        logger.info(f"Email fallback: Found matching email {email_found} on {source_url}")
                                        break
                                if email_found:
                                    break
                        except Exception as fallback_err:
                            logger.error(f"Fallback search and scrape failed for {author_name}: {fallback_err}")

                    lead = {
                        "name": author_name,
                        "email": email_found,
                        "institution": inst_name,
                        "position": profile_details.get("position", "Researcher"),
                        "research_area": concept_query,
                        "faculty_url": orcid if orcid else f"https://openalex.org/{author_id.split('/')[-1]}",
                        "recent_paper": recent_paper_title,
                        "why_selected": f"Authored highly cited paper on '{concept_query}': {recent_paper_title}",
                        "campaign": f"openalex_{concept_query.lower().replace(' ', '_')}",
                        
                        "email_source_url": source_url if email_found else None,
                        "email_source_type": source_type if email_found else "unknown",
                        "confidence": confidence,
                        "contact_permission": permission,
                        
                        "recent_publication_count": profile_details.get("recent_pub_count", 0),
                        "last_publication": pub_date or profile_details.get("last_pub", ""),
                        "active_last_3_years": profile_details.get("active_last_3", 0),
                        
                        "google_scholar_citations": profile_details.get("citations", 0),
                        "h_index": profile_details.get("h_index", 0),
                        "books_authored": 0,
                        "course_instructor": 1 if "professor" in profile_details.get("position", "").lower() else 0,
                        "newsletter": 0,
                        "youtube": 0,
                        "conference_speaker": 1 if profile_details.get("h_index", 0) > 10 else 0,
                        "editor": 1 if profile_details.get("h_index", 0) > 30 else 0,
                        "lab_director": 1 if any(kw in profile_details.get("position", "").lower() for kw in ["director", "head", "chair"]) else 0
                    }
                    
                    if email_found:
                        leads.append(lead)
                    if len(leads) >= limit:
                        break
                
                if len(leads) >= limit:
                    break
            
            # Update cursor for pagination
            next_cursor = data.get("meta", {}).get("next_cursor")
            if not next_cursor or next_cursor == cursor:
                break
            cursor = next_cursor

        return leads

    def enrich_author_profile(self, openalex_id: str) -> Dict[str, Any]:
        """Fetch additional telemetry metadata for a specific author."""
        try:
            profile = self.client.get_author_by_id(openalex_id)
            if not profile:
                return {}
            
            citations = profile.get("cited_by_count", 0)
            stats = profile.get("summary_stats", {})
            h_index = stats.get("h_index", 0)
            
            position_title = "Researcher"
            pos_info = profile.get("last_known_position", {})
            if pos_info:
                position_title = pos_info.get("title", "Researcher")

            recent_pub_count = 0
            last_pub_year = ""
            current_year = datetime.now().year
            
            counts_by_year = profile.get("counts_by_year", [])
            for c in counts_by_year:
                year = c.get("year", 0)
                works = c.get("works_count", 0)
                if current_year - 3 <= year <= current_year:
                    recent_pub_count += works
                if works > 0 and (not last_pub_year or year > int(last_pub_year)):
                    last_pub_year = str(year)
                    
            active_last_3 = 1 if recent_pub_count > 0 else 0

            return {
                "citations": citations,
                "h_index": h_index,
                "position": position_title,
                "recent_pub_count": recent_pub_count,
                "last_pub": last_pub_year,
                "active_last_3": active_last_3
            }
        except Exception as e:
            logger.error(f"Error enriching author profile {openalex_id}: {e}")
            return {}
