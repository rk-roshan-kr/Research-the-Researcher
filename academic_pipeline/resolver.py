import logging
import re
from typing import Optional
from academic_pipeline.openalex import OpenAlexClient
from academic_pipeline.semantic_scholar import SemanticScholarClient
from academic_pipeline.google_scholar import GoogleScholarScraper

logger = logging.getLogger(__name__)

class ProfileResolver:
    """
    Identifies the input query type, resolves it to the correct researcher profile
    across different academic databases (OpenAlex, Semantic Scholar, Google Scholar, ORCID),
    and merges their metadata.
    """
    def __init__(self, openalex_client: OpenAlexClient, s2_client: SemanticScholarClient, scholar_scraper: GoogleScholarScraper, search_client=None):
        self.openalex = openalex_client
        self.s2 = s2_client
        self.scholar = scholar_scraper
        self.search = search_client

    def resolve_google_scholar_profile(self, resolved_dict) -> Optional[dict]:
        """
        Attempts to autodiscover a researcher's Google Scholar profile ID,
        scrapes the profile metrics and papers, and updates the resolved dictionary.
        """
        if resolved_dict.get("google_scholar_id") and resolved_dict.get("google_scholar_data"):
            return resolved_dict.get("google_scholar_data")
            
        if not self.search:
            logger.warning("No search client available in resolver. Skipping Scholar autodiscovery.")
            return None
            
        name = resolved_dict.get("name")
        if not name:
            return None
            
        institution = resolved_dict.get("affiliation", "")
        inst_words = ""
        if institution:
            inst_words = " ".join([w.strip(",.()&") for w in institution.split() if w.lower() not in {"university", "college", "institute", "of", "and", "the", "technology", "science"}][:2])
            
        query = f'"{name}" {inst_words} site:scholar.google.com/citations'
        logger.info(f"Scholar Autodiscovery: searching Google Scholar profile with query: {query}")
        results = self.search.search(query, num_results=5)
        
        if not results and inst_words:
            fallback_query = f'"{name}" site:scholar.google.com/citations'
            logger.info(f"Scholar Autodiscovery: no results with institution, trying fallback: {fallback_query}")
            results = self.search.search(fallback_query, num_results=5)
            
        user_id = None
        for r in results:
            url = r.get("url", "")
            match = re.search(r'[?&]user=([^&]+)', url)
            if match:
                user_id = match.group(1)
                break
                
        if user_id:
            logger.info(f"Scholar Autodiscovery: Found Google Scholar ID: {user_id}")
            try:
                profile_data = self.scholar.scrape_profile(user_id)
                if profile_data:
                    resolved_dict["google_scholar_id"] = user_id
                    resolved_dict["google_scholar_data"] = profile_data
                    if not resolved_dict.get("photo_url"):
                        resolved_dict["photo_url"] = profile_data.get("photo_url")
                    return profile_data
            except Exception as e:
                logger.error(f"Failed to scrape Google Scholar profile for autodiscovered ID {user_id}: {e}")
                
        logger.info("Scholar Autodiscovery: No Google Scholar profile could be resolved.")
        return None

    def parse_input_type(self, query):
        """
        Detects what kind of input query was provided.
        Returns: tuple (query_type, cleaned_value)
        """
        query = query.strip()
        
        # 1. ORCID check
        orcid_pattern = r'(?:https?://orcid\.org/)?(\d{4}-\d{4}-\d{4}-\d{3}[0-9X])'
        match = re.search(orcid_pattern, query)
        if match:
            return "orcid", match.group(1)
            
        # 2. Google Scholar URL or ID check
        if "scholar.google" in query:
            match = re.search(r'[?&]user=([^&]+)', query)
            if match:
                return "google_scholar", match.group(1)
        # Check if it looks like a 12-char Scholar user ID (e.g. F9b9S8sAAAAJ)
        if len(query) == 12 and re.match(r'^[a-zA-Z0-9_\-]+$', query):
            return "google_scholar", query

        # 3. OpenAlex URL or ID check
        openalex_pattern = r'(?:https?://openalex\.org/)?(A\d+)'
        match = re.search(openalex_pattern, query, re.IGNORECASE)
        if match:
            return "openalex", match.group(1).upper()

        # 4. Semantic Scholar URL or ID check
        if "semanticscholar.org/author" in query:
            # e.g., https://www.semanticscholar.org/author/Oren-Etzioni/1741101
            match = re.search(r'/author/[^/]+/(\d+)|/author/(\d+)', query)
            if match:
                # Return the capture group that is not None
                return "semantic_scholar", match.group(1) or match.group(2)
        if query.isdigit():
            return "semantic_scholar", query

        # Default to name search
        return "name", query

    def resolve_researcher(self, query, interactive_callback=None):
        """
        Resolves input to a single verified set of IDs across systems.
        
        Args:
            query (str): The search input.
            interactive_callback (function): Callback for selecting among candidates if name query.
                                             Signature: fn(candidates_list) -> chosen_candidate_dict
                                             If None, auto-selects top candidate.
                                             
        Returns:
            dict: Resolved identity mapping containing:
                  {
                     'name': str,
                     'orcid': str or None,
                     'openalex_id': str or None,
                     'semantic_scholar_id': str or None,
                     'google_scholar_id': str or None,
                     'affiliation': str or None,
                     'interests': list
                  }
        """
        query_type, val = self.parse_input_type(query)
        logger.info(f"Detected input type: {query_type} with value: {val}")

        hint = None
        if query_type == "name":
            cleaned_val = val.strip()
            if " at " in cleaned_val.lower():
                match = re.search(r'\s+at\s+(.+)$', cleaned_val, re.IGNORECASE)
                if match:
                    hint = match.group(1).strip()
                    cleaned_val = cleaned_val[:match.start()].strip()
            elif " @ " in cleaned_val:
                parts = cleaned_val.split(" @ ")
                hint = parts[-1].strip()
                cleaned_val = " ".join(parts[:-1]).strip()
            else:
                match = re.search(r'\(([^)]+)\)', cleaned_val)
                if match:
                    hint = match.group(1).strip()
                    cleaned_val = re.sub(r'\([^)]+\)', '', cleaned_val).strip()
            
            if hint:
                logger.info(f"Extracted affiliation/institution hint: '{hint}' from name: '{val}'")
                val = cleaned_val

        resolved = {
            "name": "",
            "orcid": None,
            "openalex_id": None,
            "semantic_scholar_id": None,
            "google_scholar_id": None,
            "affiliation": None,
            "interests": [],
            "photo_url": None,
            "google_scholar_data": None
        }

        if query_type == "orcid":
            resolved["orcid"] = val
            # Resolve via OpenAlex first
            oa_author = self.openalex.get_author_by_orcid(val)
            if oa_author:
                resolved["openalex_id"] = oa_author.get("id", "").split("/")[-1]
                resolved["name"] = oa_author.get("display_name", "")
                resolved["affiliation"] = self._extract_oa_affiliation(oa_author)
                resolved["interests"] = [topic.get("display_name") for topic in (oa_author.get("x_concepts") or []) if topic.get("display_name")][:5]
            
            # Resolve via Semantic Scholar by searching and checking ORCID
            search_name = resolved["name"] or val
            s2_candidates = self.s2.search_authors(search_name)
            for cand in s2_candidates:
                # Retrieve full record to check external IDs
                cand_id = cand.get("authorId")
                try:
                    full_cand = self.s2.get_author_by_id(cand_id)
                    cand_orcids = ((full_cand.get("externalIds") or {}).get("ORCID") or [])
                    # Match
                    if val in cand_orcids or any(val in o for o in cand_orcids):
                        resolved["semantic_scholar_id"] = cand_id
                        if not resolved["name"]:
                            resolved["name"] = full_cand.get("name")
                        break
                except Exception as e:
                    logger.debug(f"Failed fetching details for S2 candidate {cand_id}: {e}")

        elif query_type == "google_scholar":
            resolved["google_scholar_id"] = val
            scholar_profile = self.scholar.scrape_profile(val)
            if scholar_profile:
                resolved["name"] = scholar_profile.get("name", "")
                resolved["affiliation"] = scholar_profile.get("affiliation", "")
                resolved["interests"] = (scholar_profile.get("interests") or [])
                resolved["photo_url"] = scholar_profile.get("photo_url")
                resolved["google_scholar_data"] = scholar_profile
                
                # Search OpenAlex and Semantic Scholar by name
                self._resolve_by_name_search(resolved, scholar_profile.get("name"))
            else:
                logger.warning(f"Failed to scrape Google Scholar profile for ID {val}.")
                # Fallback to resolving by ID string as if it was a name search or name search for it
                return self.resolve_researcher(val, interactive_callback)

        elif query_type == "openalex":
            resolved["openalex_id"] = val
            oa_author = self.openalex.get_author_by_id(val)
            if oa_author:
                resolved["name"] = oa_author.get("display_name", "")
                resolved["orcid"] = oa_author.get("orcid", "").split("/")[-1] if oa_author.get("orcid") else None
                resolved["affiliation"] = self._extract_oa_affiliation(oa_author)
                resolved["interests"] = [topic.get("display_name") for topic in (oa_author.get("x_concepts") or []) if topic.get("display_name")][:5]
                
                # Resolve Semantic Scholar
                if resolved["orcid"]:
                    # Try ORCID matching
                    s2_candidates = self.s2.search_authors(resolved["name"])
                    for cand in s2_candidates:
                        cand_id = cand.get("authorId")
                        try:
                            full_cand = self.s2.get_author_by_id(cand_id)
                            if resolved["orcid"] in ((full_cand.get("externalIds") or {}).get("ORCID") or []):
                                resolved["semantic_scholar_id"] = cand_id
                                break
                        except Exception:
                            pass
                
                # If still not resolved, look up by name search
                if not resolved["semantic_scholar_id"]:
                    s2_candidates = self.s2.search_authors(resolved["name"])
                    if s2_candidates:
                        # Try to find a candidate with matching affiliation
                        matched_id = None
                        oa_aff = resolved.get("affiliation") or ""
                        oa_aff_norm = re.sub(r'[^a-z0-9]', '', oa_aff.lower())
                        if oa_aff_norm:
                            for cand in s2_candidates:
                                cand_affils = cand.get("affiliations") or []
                                for aff in cand_affils:
                                    aff_name = aff.get("name", "") if isinstance(aff, dict) else aff
                                    aff_norm = re.sub(r'[^a-z0-9]', '', aff_name.lower())
                                    if aff_norm and (aff_norm in oa_aff_norm or oa_aff_norm in aff_norm):
                                        matched_id = cand.get("authorId")
                                        break
                                if matched_id:
                                    break
                        resolved["semantic_scholar_id"] = matched_id or s2_candidates[0].get("authorId")

        elif query_type == "semantic_scholar":
            resolved["semantic_scholar_id"] = val
            s2_author = self.s2.get_author_by_id(val)
            if s2_author:
                resolved["name"] = s2_author.get("name", "")
                orcids = ((s2_author.get("externalIds") or {}).get("ORCID") or [])
                resolved["orcid"] = orcids[0] if orcids else None
                
                # Affiliations in Semantic Scholar is a list
                affils = (s2_author.get("affiliations") or [])
                resolved["affiliation"] = affils[0] if affils else None
                
                # Query OpenAlex
                if resolved["orcid"]:
                    oa_author = self.openalex.get_author_by_orcid(resolved["orcid"])
                    if oa_author:
                        resolved["openalex_id"] = oa_author.get("id", "").split("/")[-1]
                        resolved["interests"] = [topic.get("display_name") for topic in (oa_author.get("x_concepts") or []) if topic.get("display_name")][:5]
                
                if not resolved["openalex_id"]:
                    oa_candidates = self.openalex.search_authors(resolved["name"])
                    if oa_candidates:
                        # Try to find a candidate with matching affiliation
                        matched_id = None
                        s2_aff = resolved.get("affiliation") or ""
                        s2_aff_norm = re.sub(r'[^a-z0-9]', '', s2_aff.lower())
                        if s2_aff_norm:
                            for cand in oa_candidates:
                                insts = cand.get("last_known_institutions", []) or cand.get("institutions", []) or []
                                for inst in insts:
                                    inst_name = inst.get("display_name", "") if isinstance(inst, dict) else inst
                                    inst_norm = re.sub(r'[^a-z0-9]', '', inst_name.lower())
                                    if inst_norm and (inst_norm in s2_aff_norm or s2_aff_norm in inst_norm):
                                        matched_id = cand.get("id", "").split("/")[-1]
                                        break
                                if matched_id:
                                    break
                        
                        chosen_cand = None
                        if matched_id:
                            for cand in oa_candidates:
                                if cand.get("id", "").split("/")[-1] == matched_id:
                                    chosen_cand = cand
                                    break
                        if not chosen_cand:
                            chosen_cand = oa_candidates[0]
                            
                        resolved["openalex_id"] = chosen_cand.get("id", "").split("/")[-1]
                        resolved["interests"] = [topic.get("display_name") for topic in (chosen_cand.get("x_concepts") or []) if topic.get("display_name")][:5]

        else: # name search
            # We search both platforms and merge for a richer candidate list
            oa_candidates = self.openalex.search_authors(val)
            s2_candidates = self.s2.search_authors(val)

            # Construct a unified candidate list for disambiguation
            oa_list = []
            for oa in oa_candidates[:10]:
                cand_id = oa.get("id", "").split("/")[-1]
                display_name = oa.get("display_name", "")
                affil = self._extract_oa_affiliation(oa)
                oa_list.append({
                    "source": "OpenAlex",
                    "id": cand_id,
                    "name": display_name,
                    "affiliation": affil,
                    "papers_count": oa.get("works_count", 0),
                    "citations_count": oa.get("cited_by_count", 0),
                    "h_index": (oa.get("summary_stats") or {}).get("h_index", 0),
                    "orcid": oa.get("orcid", "").split("/")[-1] if oa.get("orcid") else None,
                    "raw": oa
                })

            s2_list = []
            for s2 in s2_candidates[:10]:
                s2_name = s2.get("name", "")
                s2_affils = s2.get("affiliations") or []
                if s2_affils and isinstance(s2_affils[0], dict):
                    s2_affil = s2_affils[0].get("name", "")
                else:
                    s2_affil = s2_affils[0] if s2_affils else ""
                s2_list.append({
                    "source": "Semantic Scholar",
                    "id": s2.get("authorId"),
                    "name": s2_name,
                    "affiliation": s2_affil,
                    "papers_count": s2.get("paperCount", 0),
                    "citations_count": s2.get("citationCount", 0),
                    "h_index": s2.get("hIndex", 0),
                    "orcid": None,
                    "raw": s2
                })

            # Match and merge candidates from both platforms based on name & affiliation overlap
            candidates = []
            used_s2_indices = set()
            
            for oa_cand in oa_list:
                matched_s2 = None
                matched_idx = -1
                
                oa_name_norm = re.sub(r'[^a-z]', '', oa_cand["name"].lower())
                oa_aff_norm = re.sub(r'[^a-z]', '', oa_cand["affiliation"].lower())
                
                for idx, s2_cand in enumerate(s2_list):
                    if idx in used_s2_indices:
                        continue
                    
                    s2_name_norm = re.sub(r'[^a-z]', '', s2_cand["name"].lower())
                    s2_aff_norm = re.sub(r'[^a-z]', '', s2_cand["affiliation"].lower())
                    
                    # Fuzzy name match and check if names are highly related
                    name_match = (oa_name_norm == s2_name_norm) or (oa_name_norm in s2_name_norm) or (s2_name_norm in oa_name_norm)
                    
                    aff_match = False
                    if name_match:
                        if not oa_aff_norm or not s2_aff_norm:
                            # If either has no affiliation, verify by publication overlap!
                            aff_match = self._check_publication_overlap(oa_cand["id"], s2_cand["id"])
                        else:
                            # If affiliations match, it's a match. If not, check publication overlap as fallback!
                            aff_match = (oa_aff_norm in s2_aff_norm) or (s2_aff_norm in oa_aff_norm)
                            if not aff_match:
                                aff_match = self._check_publication_overlap(oa_cand["id"], s2_cand["id"])
                            
                    if name_match and aff_match:
                        matched_s2 = s2_cand
                        matched_idx = idx
                        break
                
                if matched_s2:
                    used_s2_indices.add(matched_idx)
                    candidates.append({
                        "source": "OpenAlex + S2",
                        "id": f"{oa_cand['id']} / {matched_s2['id']}",
                        "name": oa_cand["name"],
                        "affiliation": oa_cand["affiliation"] or matched_s2["affiliation"],
                        "orcid": oa_cand["orcid"],
                        "openalex_id": oa_cand["id"],
                        "semantic_scholar_id": matched_s2["id"],
                        "papers_count": max(oa_cand["papers_count"], matched_s2["papers_count"]),
                        "citations_count": max(oa_cand["citations_count"], matched_s2["citations_count"]),
                        "h_index": max(oa_cand["h_index"], matched_s2["h_index"]),
                        "raw_oa": oa_cand["raw"],
                        "raw_s2": matched_s2["raw"]
                    })
                else:
                    candidates.append({
                        "source": "OpenAlex",
                        "id": oa_cand["id"],
                        "name": oa_cand["name"],
                        "affiliation": oa_cand["affiliation"],
                        "orcid": oa_cand["orcid"],
                        "openalex_id": oa_cand["id"],
                        "semantic_scholar_id": None,
                        "papers_count": oa_cand["papers_count"],
                        "citations_count": oa_cand["citations_count"],
                        "h_index": oa_cand["h_index"],
                        "raw_oa": oa_cand["raw"],
                        "raw_s2": None
                    })
                    
            for idx, s2_cand in enumerate(s2_list):
                if idx not in used_s2_indices:
                    candidates.append({
                        "source": "Semantic Scholar",
                        "id": s2_cand["id"],
                        "name": s2_cand["name"],
                        "affiliation": s2_cand["affiliation"],
                        "orcid": None,
                        "openalex_id": None,
                        "semantic_scholar_id": s2_cand["id"],
                        "papers_count": s2_cand["papers_count"],
                        "citations_count": s2_cand["citations_count"],
                        "h_index": s2_cand["h_index"],
                        "raw_oa": None,
                        "raw_s2": s2_cand["raw"]
                    })

            # Rank candidates: prioritize affiliation hint first, then name matching similarity, then metrics as a tie-breaker
            def get_name_match_score(q_str, c_name):
                q_words = set(re.sub(r'[^a-z0-9 ]', '', q_str.lower()).split())
                c_words = set(re.sub(r'[^a-z0-9 ]', '', c_name.lower()).split())
                if not q_words:
                    return 0
                return len(q_words.intersection(c_words)) / len(q_words)

            def get_hint_match_score(c_affil):
                if not hint or not c_affil:
                    return 0
                h_words = set(re.sub(r'[^a-z0-9 ]', '', hint.lower()).split())
                a_words = set(re.sub(r'[^a-z0-9 ]', '', c_affil.lower()).split())
                intersection = h_words.intersection(a_words)
                if not intersection:
                    return 0
                return len(intersection) / len(h_words)

            candidates.sort(
                key=lambda c: (
                    get_hint_match_score(c.get("affiliation", "")),
                    get_name_match_score(val, c.get("name", "")),
                    (c.get("citations_count") or 0) +
                    (c.get("papers_count") or 0) * 10 +
                    (c.get("h_index") or 0) * 100
                ),
                reverse=True
            )

            # If interactive callback is provided and we have candidates
            if interactive_callback and candidates:
                chosen = interactive_callback(candidates)

                # Handle retry
                if isinstance(chosen, dict) and chosen.get("__action__") == "retry":
                    new_query = chosen["query"]
                    logger.info(f"User requested re-search with: {new_query!r}")
                    return self.resolve_researcher(new_query, interactive_callback)

                if chosen:
                    resolved["name"] = chosen["name"]
                    resolved["affiliation"] = chosen["affiliation"]
                    resolved["orcid"] = chosen["orcid"]
                    resolved["openalex_id"] = chosen.get("openalex_id")
                    resolved["semantic_scholar_id"] = chosen.get("semantic_scholar_id")
                    
                    raw_oa = chosen.get("raw_oa")
                    if raw_oa:
                        resolved["interests"] = [topic.get("display_name") for topic in (raw_oa.get("x_concepts") or []) if topic.get("display_name")][:5]
                    
                    # Resolve counterparts if missing
                    if resolved["openalex_id"] and not resolved["semantic_scholar_id"]:
                        s2_matches = self.s2.search_authors(chosen["name"])
                        if s2_matches:
                            if resolved["orcid"]:
                                for sm in s2_matches:
                                    try:
                                        full_sm = self.s2.get_author_by_id(sm.get("authorId"))
                                        if resolved["orcid"] in ((full_sm.get("externalIds") or {}).get("ORCID") or []):
                                            resolved["semantic_scholar_id"] = sm.get("authorId")
                                            break
                                    except Exception:
                                        pass
                            if not resolved["semantic_scholar_id"]:
                                matched_id = None
                                oa_aff = resolved.get("affiliation") or ""
                                oa_aff_norm = re.sub(r'[^a-z0-9]', '', oa_aff.lower())
                                if oa_aff_norm:
                                    for sm in s2_matches:
                                        cand_affils = sm.get("affiliations") or []
                                        for aff in cand_affils:
                                            aff_name = aff.get("name", "") if isinstance(aff, dict) else aff
                                            aff_norm = re.sub(r'[^a-z0-9]', '', aff_name.lower())
                                            if aff_norm and (aff_norm in oa_aff_norm or oa_aff_norm in aff_norm):
                                                matched_id = sm.get("authorId")
                                                break
                                        if matched_id:
                                            break
                                resolved["semantic_scholar_id"] = matched_id or s2_matches[0].get("authorId")
                                
                    elif resolved["semantic_scholar_id"] and not resolved["openalex_id"]:
                        try:
                            full_s2 = self.s2.get_author_by_id(resolved["semantic_scholar_id"])
                            orcids = ((full_s2.get("externalIds") or {}).get("ORCID") or [])
                            resolved["orcid"] = orcids[0] if orcids else None
                        except Exception:
                            pass
                        if resolved["orcid"]:
                            oa_rec = self.openalex.get_author_by_orcid(resolved["orcid"])
                            if oa_rec:
                                resolved["openalex_id"] = oa_rec.get("id", "").split("/")[-1]
                                resolved["interests"] = [t.get("display_name") for t in (oa_rec.get("x_concepts") or []) if t.get("display_name")][:5]
                        if not resolved["openalex_id"]:
                            oa_matches = self.openalex.search_authors(chosen["name"])
                            if oa_matches:
                                matched_id = None
                                s2_aff = resolved.get("affiliation") or ""
                                s2_aff_norm = re.sub(r'[^a-z0-9]', '', s2_aff.lower())
                                if s2_aff_norm:
                                    for om in oa_matches:
                                        insts = om.get("last_known_institutions", []) or om.get("institutions", []) or []
                                        for inst in insts:
                                            inst_name = inst.get("display_name", "") if isinstance(inst, dict) else inst
                                            inst_norm = re.sub(r'[^a-z0-9]', '', inst_name.lower())
                                            if inst_norm and (inst_norm in s2_aff_norm or s2_aff_norm in inst_norm):
                                                matched_id = om.get("id", "").split("/")[-1]
                                                break
                                        if matched_id:
                                            break
                                chosen_om = None
                                if matched_id:
                                    for om in oa_matches:
                                        if om.get("id", "").split("/")[-1] == matched_id:
                                            chosen_om = om
                                            break
                                if not chosen_om:
                                    chosen_om = oa_matches[0]
                                resolved["openalex_id"] = chosen_om.get("id", "").split("/")[-1]
                                resolved["interests"] = [t.get("display_name") for t in (chosen_om.get("x_concepts") or []) if t.get("display_name")][:5]

                    self.resolve_google_scholar_profile(resolved)
                    return resolved

            # Auto-select path
            if candidates:
                top_cand = candidates[0]
                resolved["name"] = top_cand["name"]
                resolved["affiliation"] = top_cand["affiliation"]
                resolved["orcid"] = top_cand["orcid"]
                resolved["openalex_id"] = top_cand.get("openalex_id")
                resolved["semantic_scholar_id"] = top_cand.get("semantic_scholar_id")
                
                raw_oa = top_cand.get("raw_oa")
                if raw_oa:
                    resolved["interests"] = [topic.get("display_name") for topic in (raw_oa.get("x_concepts") or []) if topic.get("display_name")][:5]
                
                if resolved["openalex_id"] and not resolved["semantic_scholar_id"]:
                    s2_matches = self.s2.search_authors(top_cand["name"])
                    if s2_matches:
                        if resolved["orcid"]:
                            for sm in s2_matches:
                                try:
                                    full_sm = self.s2.get_author_by_id(sm.get("authorId"))
                                    if resolved["orcid"] in ((full_sm.get("externalIds") or {}).get("ORCID") or []):
                                        resolved["semantic_scholar_id"] = sm.get("authorId")
                                        break
                                except Exception:
                                    pass
                        if not resolved["semantic_scholar_id"]:
                            matched_id = None
                            oa_aff = resolved.get("affiliation") or ""
                            oa_aff_norm = re.sub(r'[^a-z0-9]', '', oa_aff.lower())
                            if oa_aff_norm:
                                for sm in s2_matches:
                                    cand_affils = sm.get("affiliations") or []
                                    for aff in cand_affils:
                                        aff_name = aff.get("name", "") if isinstance(aff, dict) else aff
                                        aff_norm = re.sub(r'[^a-z0-9]', '', aff_name.lower())
                                        if aff_norm and (aff_norm in oa_aff_norm or oa_aff_norm in aff_norm):
                                            matched_id = sm.get("authorId")
                                            break
                                    if matched_id:
                                        break
                            resolved["semantic_scholar_id"] = matched_id or s2_matches[0].get("authorId")
                
                elif resolved["semantic_scholar_id"] and not resolved["openalex_id"]:
                    try:
                        full_s2 = self.s2.get_author_by_id(resolved["semantic_scholar_id"])
                        orcids = ((full_s2.get("externalIds") or {}).get("ORCID") or [])
                        resolved["orcid"] = orcids[0] if orcids else None
                    except Exception:
                        pass
                    if resolved["orcid"]:
                        oa_rec = self.openalex.get_author_by_orcid(resolved["orcid"])
                        if oa_rec:
                            resolved["openalex_id"] = oa_rec.get("id", "").split("/")[-1]
                            resolved["interests"] = [t.get("display_name") for t in (oa_rec.get("x_concepts") or []) if t.get("display_name")][:5]
                    if not resolved["openalex_id"]:
                        oa_matches = self.openalex.search_authors(top_cand["name"])
                        if oa_matches:
                            matched_id = None
                            s2_aff = resolved.get("affiliation") or ""
                            s2_aff_norm = re.sub(r'[^a-z0-9]', '', s2_aff.lower())
                            if s2_aff_norm:
                                for om in oa_matches:
                                    insts = om.get("last_known_institutions", []) or om.get("institutions", []) or []
                                    for inst in insts:
                                        inst_name = inst.get("display_name", "") if isinstance(inst, dict) else inst
                                        inst_norm = re.sub(r'[^a-z0-9]', '', inst_name.lower())
                                        if inst_norm and (inst_norm in s2_aff_norm or s2_aff_norm in inst_norm):
                                            matched_id = om.get("id", "").split("/")[-1]
                                            break
                                    if matched_id:
                                        break
                            chosen_om = None
                            if matched_id:
                                for om in oa_matches:
                                    if om.get("id", "").split("/")[-1] == matched_id:
                                        chosen_om = om
                                        break
                            if not chosen_om:
                                chosen_om = oa_matches[0]
                            resolved["openalex_id"] = chosen_om.get("id", "").split("/")[-1]
                            resolved["interests"] = [t.get("display_name") for t in (chosen_om.get("x_concepts") or []) if t.get("display_name")][:5]

        self.resolve_google_scholar_profile(resolved)
        return resolved

    def _resolve_by_name_search(self, resolved_dict, name):
        """Helper to find IDs on OpenAlex and Semantic Scholar given a name."""
        if not name:
            return
            
        # 1. OpenAlex search
        try:
            oa_cand = self.openalex.search_authors(name)
            if oa_cand:
                # Find best candidate (exact name match or matching affiliation keyword)
                best_oa = oa_cand[0]
                # Compare affiliations if available
                scholar_affil = resolved_dict.get("affiliation", "").lower()
                for cand in oa_cand[:3]:
                    cand_affil = self._extract_oa_affiliation(cand).lower()
                    # Check if there is affiliation overlap
                    if scholar_affil and cand_affil and (scholar_affil in cand_affil or cand_affil in scholar_affil):
                        best_oa = cand
                        break
                resolved_dict["openalex_id"] = best_oa.get("id", "").split("/")[-1]
                resolved_dict["orcid"] = best_oa.get("orcid", "").split("/")[-1] if best_oa.get("orcid") else None
                resolved_dict["interests"] = [t.get("display_name") for t in (best_oa.get("x_concepts") or []) if t.get("display_name")][:5]
        except Exception as e:
            logger.warning(f"Error searching OpenAlex by name during cross-reference: {e}")

        # 2. Semantic Scholar search
        try:
            s2_cand = self.s2.search_authors(name)
            if s2_cand:
                best_s2 = s2_cand[0]
                # If we have an ORCID, try to match exactly
                if resolved_dict["orcid"]:
                    for cand in s2_cand[:3]:
                        try:
                            full_cand = self.s2.get_author_by_id(cand.get("authorId"))
                            if resolved_dict["orcid"] in ((full_cand.get("externalIds") or {}).get("ORCID") or []):
                                best_s2 = cand
                                break
                        except Exception:
                            pass
                resolved_dict["semantic_scholar_id"] = best_s2.get("authorId")
        except Exception as e:
            logger.warning(f"Error searching Semantic Scholar by name during cross-reference: {e}")

    def _extract_oa_affiliation(self, oa_author_record):
        """Extract institution display name from OpenAlex author record."""
        last_affils = (oa_author_record.get("last_known_institutions") or [])
        if last_affils:
            return last_affils[0].get("display_name", "")
            
        # Fallback to legacy field format
        affils = (oa_author_record.get("affiliations") or [])
        if affils:
            inst = affils[0].get("institution", {})
            if inst:
                return inst.get("display_name", "")
        return ""

    def _check_publication_overlap(self, oa_id: str, s2_id: str) -> bool:
        """
        Fetches top 5 recent papers from both systems and checks for title overlaps to cross-verify identity.
        Returns True if at least one paper matches (Jaccard similarity of trigrams >= 0.80 or direct match).
        """
        if not oa_id or not s2_id:
            return False
        try:
            logger.info(f"Cross-verifying publication overlap between OpenAlex:{oa_id} and SemanticScholar:{s2_id}")
            oa_works = self.openalex.get_author_works(oa_id, max_works=5)
            s2_papers = self.s2.get_author_papers(s2_id, max_papers=5)
            
            oa_titles = [re.sub(r'[^a-z0-9]', '', w.get("title", "").lower()) for w in oa_works if w.get("title")]
            s2_titles = [re.sub(r'[^a-z0-9]', '', p.get("title", "").lower()) for p in s2_papers if p.get("title")]
            
            if not oa_titles or not s2_titles:
                return False
                
            for oa_t in oa_titles:
                if len(oa_t) < 10:
                    continue
                for s2_t in s2_titles:
                    if len(s2_t) < 10:
                        continue
                    if oa_t == s2_t or oa_t in s2_t or s2_t in oa_t:
                        logger.info("Found exact/substring publication title overlap!")
                        return True
                    # Trigram Jaccard match for fuzzy matching
                    t1 = set(oa_t[i:i+3] for i in range(len(oa_t)-2))
                    t2 = set(s2_t[i:i+3] for i in range(len(s2_t)-2))
                    if t1 and t2:
                        jaccard = len(t1.intersection(t2)) / len(t1.union(t2))
                        if jaccard >= 0.80:
                            logger.info(f"Found fuzzy publication title overlap (Jaccard={jaccard:.2f})!")
                            return True
            return False
        except Exception as e:
            logger.debug(f"Failed to check publication overlap: {e}")
            return False

