"""
university_discovery.py — Relevance-first, Program-centric University Discovery Engine.
"""

from __future__ import annotations

import os
import time
import logging
import asyncio
import numpy as np
from typing import Optional, Literal

from academic_pipeline.openalex import OpenAlexClient
from academic_pipeline.cache_manager import CacheManager
from academic_pipeline.investigation_engine import InvestigationEngine
from academic_pipeline.collaboration_scorer import CollaborationScorer
from academic_pipeline.investigation_schemas import ProfessorMemory, MemoryEvent
from academic_pipeline.local_llm import LocalLLMClient
from academic_pipeline.embedding_cache import cached_encode
from academic_pipeline.profile_cache import ProfileCache

logger = logging.getLogger(__name__)

# Predefined concept IDs mapping for relevance-first OpenAlex query
CONCEPT_IDS = {
    # EarthOS domains
    "computer science": "C41008148",
    "artificial intelligence": "C154945302",
    "cognitive science": "C169900460",
    "neuroscience": "C169760540",
    "robotics": "C50522688",
    "machine learning": "C119857082",
    # TARS domains
    "physics": "C121332964",
    "astrophysics": "C635677",
    "astronomy": "C205649164",
}

def reconstruct_abstract(inverted_index: Optional[dict]) -> str:
    """Reconstruct abstract string from OpenAlex abstract_inverted_index."""
    if not inverted_index:
        return ""
    words = {}
    for word, positions in inverted_index.items():
        for pos in positions:
            words[pos] = word
    sorted_positions = sorted(words.keys())
    return " ".join(words[pos] for pos in sorted_positions)

def extract_shared_themes(text1: str, text2: str, self_model: dict) -> list[str]:
    """
    Extract shared themes/topics between two papers using key term matching.
    """
    import re
    t1 = text1.lower()
    t2 = text2.lower()
    
    user_topics = self_model.get("topics", [])
    user_methods = self_model.get("methods", []) or self_model.get("methods_used", []) or []
    candidates = list(user_topics) + list(user_methods)
    
    common_concepts = [
        "continual learning", "curriculum learning", "curriculum ordering", "representation learning",
        "latent space", "latent dynamics", "dimensionality reduction", "representation geometry",
        "cognitive validation", "causal discovery", "neural analysis", "behavioral analysis",
        "computational neuroscience", "cognitive architecture", "developmental psychology",
        "representational similarity", "cka", "rsa", "cebra", "deep learning", "coherence"
    ]
    for c in common_concepts:
        if c not in candidates:
            candidates.append(c)
            
    shared = []
    for cand in candidates:
        if not cand.strip():
            continue
        cand_lower = cand.lower()
        pattern = r'\b' + re.escape(cand_lower) + r'\b'
        if re.search(pattern, t1) and re.search(pattern, t2):
            if cand_lower == "cka":
                shared.append("CKA representational similarity")
            elif cand_lower == "rsa":
                shared.append("RSA representational similarity")
            elif cand_lower == "cebra":
                shared.append("CEBRA embeddings")
            else:
                shared.append(cand.strip())
                
    seen = set()
    unique_shared = []
    for s in shared:
        s_low = s.lower()
        if s_low not in seen:
            seen.add(s_low)
            unique_shared.append(s)
            
    if not unique_shared:
        general_terms = ["learning", "representation", "cognition", "model", "network", "neural", "behavior", "memory", "dynamics"]
        for gt in general_terms:
            pattern = r'\b' + re.escape(gt) + r'\b'
            if re.search(pattern, t1) and re.search(pattern, t2):
                unique_shared.append(gt.capitalize())
                
    return unique_shared[:3]

class UniversityDiscoveryEngine:
    """
    Discovers potential research collaborators at a specific university.
    Optimizes for research relevance-first, preventing prestige and citation bias.
    """

    def __init__(
        self,
        cache_manager: CacheManager,
        investigation_engine: InvestigationEngine,
        llm_client: Optional[LocalLLMClient] = None,
        email: str = "researcher-pipeline@example.com"
    ):
        self._cache = cache_manager
        self._inv_engine = investigation_engine
        self._llm = llm_client or LocalLLMClient()
        self._openalex = OpenAlexClient(email=email)
        self._scorer = CollaborationScorer(cache_manager=cache_manager, llm_client=self._llm)
        from academic_pipeline.research_graph import ResearchProgramGraph
        self._graph = ResearchProgramGraph(cache_dir=cache_manager.cache_dir)

    def discover(
        self,
        university_name: str,
        mode: Literal["research", "internship"] = "internship",
        limit: int = 20,
        model_name: str = "qwen2.5-coder:7b"
    ) -> str:
        """Synchronous wrapper for discover_async to maintain backward compatibility."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                return ex.submit(asyncio.run, self.discover_async(university_name, mode, limit, model_name)).result()
        else:
            return asyncio.run(self.discover_async(university_name, mode, limit, model_name))

    async def discover_async(
        self,
        university_name: str,
        mode: Literal["research", "internship"] = "internship",
        limit: int = 20,
        model_name: str = "qwen2.5-coder:7b"
    ) -> str:
        """
        Executes the university discovery flow asynchronously.
        
        Args:
            university_name (str): Target university (e.g. "EPFL").
            mode (str): "research" (Mode A: alignment only) or "internship" (Mode B: alignment + accessibility).
            limit (int): Cap on candidate profiles returned (default 20).
            model_name (str): LLM model name for tension detection.

        Returns:
            str: Path to the generated Markdown report.
        """
        logger.info(f"Starting university discovery for '{university_name}' in '{mode}' mode (limit={limit})")
        
        # 1. Resolve institution via OpenAlex
        inst_results = self._openalex.search_institutions(university_name)
        if not inst_results:
            raise ValueError(f"Could not resolve institution for '{university_name}' via OpenAlex.")
        
        inst = inst_results[0]
        inst_id = inst.get("id")
        inst_display_name = inst.get("display_name", university_name)
        logger.info(f"Resolved university: {inst_display_name} [{inst_id}]")

        # 1.5. Crawl and index complete faculty list from OpenAlex
        UNIVERSITY_ALIASES = {
            "École Polytechnique Fédérale de Lausanne": "EPFL",
            "cole Polytechnique Fdrale de Lausanne": "EPFL",
            "Epfl": "EPFL",
            "Swiss Federal Institute of Technology Lausanne": "EPFL",
            "Massachusetts Institute of Technology": "MIT",
            "Stanford University": "Stanford",
            "Harvard University": "Harvard",
            "University of California, Berkeley": "UC Berkeley",
            "California Institute of Technology": "Caltech",
            "ETH Zurich": "ETH Zurich",
            "Eidgenössische Technische Hochschule Zürich": "ETH Zurich",
        }
        alias = UNIVERSITY_ALIASES.get(inst_display_name, inst_display_name[:30] + "..." if len(inst_display_name) > 30 else inst_display_name)

        logger.info(f"Crawling complete researcher list for {inst_display_name} (alias: {alias}) from OpenAlex...")
        faculty_list = await asyncio.to_thread(
            self._openalex.get_institution_authors, inst_id, 5
        )
        logger.info(f"Crawled {len(faculty_list)} researchers with h_index > 5 from OpenAlex.")
        self._cache.save_crawled_faculty(inst_display_name, alias, faculty_list)
        logger.info(f"Indexed complete researcher list for {inst_display_name} in the local database.")


        # 2. Identify target concepts based on active investigations
        active_invs = [inv for inv in self._inv_engine.get_all_investigations() if inv.status in ("active", "blocked")]
        target_concepts = set()
        
        if active_invs:
            all_topics_text = " ".join(
                (inv.title + " " + inv.description + " " + " ".join(inv.related_topics)).lower()
                for inv in active_invs
            )
            
            # Match TARS
            if any(w in all_topics_text for w in ["tars", "exoplanet", "astrophysics", "astronomy", "transit", "physics"]):
                target_concepts.update([CONCEPT_IDS["physics"], CONCEPT_IDS["astrophysics"], CONCEPT_IDS["astronomy"]])
            
            # Match EarthOS
            if any(w in all_topics_text for w in ["earthos", "cognitive", "agent", "memory", "learning", "robot", "contradiction", "continual"]):
                target_concepts.update([
                    CONCEPT_IDS["computer science"],
                    CONCEPT_IDS["artificial intelligence"],
                    CONCEPT_IDS["cognitive science"],
                    CONCEPT_IDS["neuroscience"],
                    CONCEPT_IDS["robotics"],
                    CONCEPT_IDS["machine learning"]
                ])
                
        # Default fallback
        if not target_concepts:
            target_concepts.update([
                CONCEPT_IDS["computer science"],
                CONCEPT_IDS["artificial intelligence"],
                CONCEPT_IDS["cognitive science"],
                CONCEPT_IDS["neuroscience"],
                CONCEPT_IDS["astrophysics"],
                CONCEPT_IDS["machine learning"]
            ])

        concept_list = list(target_concepts)
        logger.info(f"Filtering authors by OpenAlex concepts: {concept_list}")

        # 3. Retrieve authors
        authors = self._openalex.get_authors_by_institution_and_concepts(inst_id, concept_list, max_authors=80)
        logger.info(f"Retrieved {len(authors)} candidate authors from OpenAlex")

        if not authors:
            raise ValueError("No authors found matching the institution and concept filter.")

        # 4. Fast Screening using Active Blocker Embeddings
        active_blockers = []
        for inv in active_invs:
            active_blockers.extend(inv.open_blockers)

        self_model = self._cache.get_self_model() or {}
        
        blocker_embs = []
        for b in active_blockers:
            if b.embedding:
                blocker_embs.append(np.array(b.embedding))
            else:
                emb = cached_encode([b.description], model_name="BAAI/bge-large-en-v1.5", normalize_embeddings=True)[0]
                b.embedding = emb.tolist()
                blocker_embs.append(emb)

        if not blocker_embs and self_model.get("topic_embedding"):
            blocker_embs.append(np.array(self_model["topic_embedding"]))

        # Asynchronous screening wrapper (MAX_CONCURRENT_PROFESSORS = 8)
        sem_prof = asyncio.Semaphore(8)
        
        async def screen_one_author(author):
            author_id = author.get("id")
            async with sem_prof:
                works = await asyncio.to_thread(self._openalex.get_author_works, author_id, max_works=20)
            if not works:
                return None
            work_titles = [w.get("title") for w in works if w.get("title")]
            if not work_titles:
                return None
            
            work_embs = await asyncio.to_thread(
                cached_encode, work_titles, model_name="BAAI/bge-large-en-v1.5", normalize_embeddings=True
            )
            
            max_sim = 0.0
            if blocker_embs:
                for blocker_emb in blocker_embs:
                    sims = work_embs @ blocker_emb
                    max_sim = max(max_sim, float(np.max(sims)))
            
            return {
                "author": author,
                "works": works,
                "screen_score": max_sim
            }

        logger.info("Screening PIs concurrently...")
        screen_tasks = [screen_one_author(a) for a in authors]
        screened_results = await asyncio.gather(*screen_tasks)
        screened_candidates = [r for r in screened_results if r is not None]

        # Sort by relevance score descending
        screened_candidates.sort(key=lambda x: x["screen_score"], reverse=True)
        top_candidates = screened_candidates[:limit]
        logger.info(f"Fast screening completed. Running deep program scoring for top {len(top_candidates)} PIs.")

        # 5. Deep Scoring, Program Clustering, and Dossier Construction
        profile_cache = ProfileCache(cache_dir=self._cache.cache_dir)
        all_program_fits = []

        async def process_pi(cand):
            author = cand["author"]
            works = cand["works"]
            author_id = author.get("id")
            author_name = author.get("display_name", "Unknown")
            
            # Check cached profile database
            cached = profile_cache.get_profile(author_id)
            if cached:
                dossier = cached["dossier"]
                programs = cached["programs"]
            else:
                h_index = author.get("h_index") or author.get("summary_stats", {}).get("h_index") or 0
                career_stage = "assistant" if h_index < 15 else ("associate" if h_index < 30 else "full")
                pub_years = [w.get("publication_year") for w in works if w.get("publication_year")]
                last_pub_year = max(pub_years) if pub_years else None
                co_author_counts = [len(w.get("authorships", [])) for w in works]
                avg_co_authors = sum(co_author_counts) / len(co_author_counts) if co_author_counts else 0
                student_supervision_signal = min(1.0, avg_co_authors / 4.0)

                dossier = {
                    "name": author_name,
                    "openalex_id": author_id,
                    "institution": inst_display_name,
                    "position": f"{career_stage.capitalize()} Professor",
                    "h_index": h_index,
                    "citationCount": author.get("cited_by_count", 0),
                    "career_stage": career_stage,
                    "last_publication_year": last_pub_year,
                    "student_supervision_signal": student_supervision_signal,
                    "publications": [
                        {
                            "title": w.get("title", ""),
                            "doi": w.get("doi") or f"no_doi_{author_id}_{i}",
                            "abstract": reconstruct_abstract(w.get("abstract_inverted_index")),
                            "year": w.get("publication_year"),
                            "citations": w.get("cited_by_count", 0),
                            "venue": w.get("primary_location", {}).get("source", {}).get("display_name") if w.get("primary_location") and w.get("primary_location").get("source") else None,
                        }
                        for i, w in enumerate(works)
                    ]
                }
                programs = await self._extract_programs_for_dossier(dossier, model_name)
                profile_cache.save_profile(author_id, dossier, programs)

            prof_program_fits = []
            for prog in programs:
                program_papers = prog["publications"]
                
                # Construct sub-dossier representing only this program
                prog_dossier = {
                    **dossier,
                    "publications": program_papers,
                    "last_publication_year": max((p.get("year") for p in program_papers if p.get("year")), default=dossier["last_publication_year"]),
                    "program_claims": prog.get("claims", []),
                    "program_methods": prog.get("methods", []),
                    "program_datasets": prog.get("datasets", []),
                    "program_metrics": prog.get("metrics", []),
                    "program_open_questions": prog.get("open_questions", []),
                    "program_title": prog.get("title", ""),
                    "program_summary": prog.get("summary", ""),
                }
                
                # Compute program-level scores
                score_res = self._scorer.compute_score(self_model, active_invs, prog_dossier, generate_hypotheses=False)

                # Save program and claims to graph database
                try:
                    import hashlib
                    from academic_pipeline.research_graph import ResearchProgramNode, ClaimNode, EvidenceNeedNode
                    
                    prog_id = "prog_" + hashlib.sha256(f"{author_id}_{prog['title']}".encode()).hexdigest()[:16]
                    prog_node = ResearchProgramNode(
                        id=prog_id,
                        title=prog["title"],
                        summary=prog["summary"],
                        university=inst_display_name,
                        professor_name=author_name,
                        last_refresh=time.time()
                    )
                    self._graph.save_program(prog_node)

                    # Save claims
                    for c in prog.get("claims", []):
                        c_id = "claim_" + hashlib.sha256(f"{prog_id}_{c['claim']}".encode()).hexdigest()[:16]
                        claim_node = ClaimNode(
                            id=c_id,
                            text=c["claim"],
                            type="theirs",
                            confidence=c.get("confidence", 1.0),
                            paper_doi=c.get("paper_doi", ""),
                            method=c.get("method", "N/A"),
                            dataset=c.get("dataset", "N/A"),
                            metric=c.get("metric", "N/A"),
                            program_id=prog_id
                        )
                        self._graph.save_claim(claim_node)

                    # Save evidence needs and probabilities
                    for m in score_res.get("resolution_matches", []):
                        matching_blocker = None
                        for inv in active_invs:
                            for b in inv.blockers:
                                if b.id == m["blocker_id"]:
                                    matching_blocker = b
                                    break
                            if matching_blocker:
                                break
                        
                        req_method = matching_blocker.required_method if matching_blocker else "N/A"
                        req_dataset = matching_blocker.required_dataset if matching_blocker else "N/A"
                        req_metric = matching_blocker.required_metric if matching_blocker else "N/A"

                        need_node = EvidenceNeedNode(
                            id=f"need_{m['blocker_id']}",
                            blocker_id=m["blocker_id"],
                            investigation_id=m["investigation_id"],
                            description=m["blocker_description"],
                            method=req_method,
                            dataset=req_dataset,
                            metric=req_metric
                        )
                        self._graph.save_evidence_need(need_node)

                        self._graph.save_evidence_probability(
                            need_id=f"need_{m['blocker_id']}",
                            program_id=prog_id,
                            probability=m["confidence"],
                            rationale=m.get("rationale", "")
                        )
                except Exception as e:
                    logger.warning(f"Could not save program and claims to graph database: {e}")
                
                # Persistence to CRM database memory
                prof_mem_dict = self._cache.get_professor_memory(author_id)
                if prof_mem_dict:
                    prof_mem = ProfessorMemory.from_dict(prof_mem_dict)
                else:
                    prof_mem = ProfessorMemory.new(author_name, inst_display_name, author_id)
                
                prof_mem.h_index = dossier["h_index"]
                prof_mem.career_stage = dossier["career_stage"]
                prof_mem.last_publication_year = dossier["last_publication_year"]
                prof_mem.student_supervision_signal = dossier["student_supervision_signal"]
                
                # Keep tracking highest program fit in CRM
                if prof_mem.last_score is None or score_res.get("total_score", 0.0) > prof_mem.last_score:
                    prof_mem.last_score = score_res.get("total_score", 0.0)
                    
                lab_accessibility, _ = self._scorer._score_lab_accessibility(prog_dossier)
                readiness = prof_mem.get_collaboration_readiness_score(lab_accessibility)
                self._cache.save_professor_memory(author_id, prof_mem.to_dict())

                dims = score_res.get("dimensions", {})
                enrs = dims.get("enrs", {}).get("contribution", 0.0)
                topic = dims.get("topic_overlap", {}).get("contribution", 0.0)
                gap = dims.get("gap_complement", {}).get("contribution", 0.0)
                method = dims.get("method_compat", {}).get("contribution", 0.0)
                lab_access = dims.get("lab_access", {}).get("contribution", 0.0)
                temporal = dims.get("temporal_act", {}).get("contribution", 0.0)

                research_utility = round(((enrs + topic + gap + method) / 85.0) * 100.0, 2)
                accessibility_score = round(((lab_access + temporal) / 15.0) * 100.0, 2)

                if research_utility >= 75:
                    contact_timing = "Now" if accessibility_score >= 50 else "After Experiment"
                else:
                    contact_timing = "After Paper" if accessibility_score >= 50 else "Monitor"

                if mode == "research":
                    final_score = research_utility
                else:
                    final_score = round((0.70 * research_utility) + (30.0 * (accessibility_score / 100.0)), 2)

                # Overlaps
                self_topics = self_model.get("topics", [])
                top_concepts = [c.get("display_name") for c in author.get("x_concepts", [])][:10]
                literal_overlap = [t for t in self_topics if any(t.lower() == pc.lower() for pc in top_concepts)]
                
                semantic_overlap = []
                if self_topics and top_concepts:
                    try:
                        self_embs = cached_encode(self_topics, model_name="BAAI/bge-large-en-v1.5", normalize_embeddings=True)
                        prof_embs = cached_encode(top_concepts, model_name="BAAI/bge-large-en-v1.5", normalize_embeddings=True)
                        sim_matrix = self_embs @ prof_embs.T
                        for i, self_t in enumerate(self_topics):
                             if self_t in literal_overlap:
                                 continue
                             best_idx = int(np.argmax(sim_matrix[i]))
                             best_sim = float(sim_matrix[i, best_idx])
                             if best_sim >= 0.40:
                                 semantic_overlap.append({
                                     "user_topic": self_t,
                                     "prof_topic": top_concepts[best_idx],
                                     "similarity": best_sim
                                 })
                    except Exception as e:
                        logger.warning(f"Semantic overlap error: {e}")

                # Paper-level overlaps
                paper_overlaps = []
                user_pubs = [p for p in self_model.get("publication_vectors", []) if p.get("embedding")]
                prof_paper_embs = score_res.get("paper_embeddings", [])
                prof_paper_meta = score_res.get("paper_meta", [])

                if user_pubs and prof_paper_embs:
                    try:
                        user_embs = np.array([p["embedding"] for p in user_pubs])
                        prof_embs = np.array(prof_paper_embs)
                        sim_matrix = user_embs @ prof_embs.T

                        for u_idx, u_pub in enumerate(user_pubs):
                            for p_idx, p_meta in enumerate(prof_paper_meta):
                                sim = float(sim_matrix[u_idx, p_idx])
                                if sim >= 0.30:
                                    shared_themes = extract_shared_themes(
                                        u_pub.get("title", "") + " " + u_pub.get("abstract_snippet", ""),
                                        p_meta.get("title", "") + " " + p_meta.get("abstract", ""),
                                        self_model
                                    )
                                    paper_overlaps.append({
                                        "user_paper_title": u_pub.get("title", ""),
                                        "user_paper_doi": u_pub.get("doi", ""),
                                        "prof_paper_title": p_meta.get("title", ""),
                                        "prof_paper_doi": p_meta.get("doi", ""),
                                        "similarity": sim,
                                        "shared_themes": shared_themes
                                    })
                        paper_overlaps.sort(key=lambda x: x["similarity"], reverse=True)
                    except Exception as e:
                        logger.warning(f"Paper overlaps error: {e}")

                # Contradiction Detection between our claims and their program claims
                prog_contradictions = []
                try:
                    from academic_pipeline.nli_filter import NLIFilter
                    nli = NLIFilter()
                    my_claims = self_model.get("structured_claims", [])
                    their_claims = prog.get("claims", [])
                    
                    candidate_pairs = nli.find_contradiction_candidates(my_claims, their_claims)
                    for my_c, their_c in candidate_pairs[:5]:
                        analysis = await self._llm.analyze_contradiction_details(my_c, their_c)
                        if analysis.get("is_contradiction", False):
                            from academic_pipeline.research_graph import ContradictionLink
                            import hashlib
                            link_id = "contradiction_" + hashlib.sha256(f"{my_c['id']}_{their_c['id']}".encode()).hexdigest()[:16]
                            link = ContradictionLink(
                                id=link_id,
                                my_claim_id=my_c["id"],
                                their_claim_id=their_c["id"],
                                type=analysis.get("contradiction_type", "empirical"),
                                resolution_hypothesis=analysis.get("resolution_hypothesis", ""),
                                severity=analysis.get("severity", 0.5),
                                confidence=analysis.get("confidence", 0.5),
                                status="open"
                            )
                            self._graph.save_contradiction(link)
                            
                            prog_contradictions.append({
                                "id": link_id,
                                "my_claim": my_c["claim"],
                                "my_claim_doi": my_c.get("paper_doi", ""),
                                "their_claim": their_c["claim"],
                                "their_claim_doi": their_c.get("paper_doi", ""),
                                "type": link.type,
                                "resolution_hypothesis": link.resolution_hypothesis,
                                "severity": link.severity,
                                "confidence": link.confidence,
                                "status": link.status
                            })
                except Exception as e:
                    logger.warning(f"Could not run contradiction detection for program: {e}")

                prof_program_fits.append({
                    "prof_id": author_id,
                    "name": author_name,
                    "position": dossier["position"],
                    "h_index": dossier["h_index"],
                    "last_pub_year": dossier["last_publication_year"],
                    "student_supervision_signal": dossier["student_supervision_signal"],
                    "alignment_score": score_res.get("total_score", 0.0),
                    "final_score": final_score,
                    "research_utility": research_utility,
                    "accessibility_score": accessibility_score,
                    "contact_timing": contact_timing,
                    "paper_overlaps": paper_overlaps,
                    "dimensions": score_res.get("dimensions", {}),
                    "best_match": max(score_res.get("resolution_matches", []), key=lambda m: m.get("similarity", 0), default=None),
                    "recommended_action": "Draft email" if final_score > 70 else "Monitor",
                    "research_areas": top_concepts,
                    "literal_overlap": literal_overlap,
                    "semantic_overlap": semantic_overlap,
                    "publications": program_papers,
                    "program_title": prog["title"],
                    "program_summary": prog["summary"],
                    "claims": prog.get("claims", []),
                    "contradictions": prog_contradictions,
                    "all_resolution_matches": score_res.get("resolution_matches", [])
                })
            return prof_program_fits

        logger.info("Scoring and clustering research programs...")
        tasks = [process_pi(cand) for cand in top_candidates]
        flat_results = await asyncio.gather(*tasks)
        all_program_fits = [fit for res in flat_results for fit in res]

        # Rank all programs across all professors
        all_program_fits.sort(key=lambda x: x["final_score"], reverse=True)
        top_programs = all_program_fits[:limit]

        # Load self-model context to help the LLM write sharper internal annotations
        my_work_context = None
        try:
            sm_data = self._cache.get_self_model()
            if sm_data:
                topics = ", ".join(sm_data.get("topics", [])[:6])
                strong_claims = "\n".join(
                    f"- {c['text'] if isinstance(c, dict) else c}"
                    for c in sm_data.get("strong_claims", [])[:4]
                )
                open_questions = "\n".join(
                    f"- {q['text'] if isinstance(q, dict) else q}"
                    for q in sm_data.get("open_questions", [])[:3]
                )
                my_work_context = (
                    f"Research Topics: {topics}\n"
                    f"Key Claims:\n{strong_claims}\n"
                    f"Open Questions:\n{open_questions}\n"
                )
        except Exception as e:
            logger.warning(f"Could not load self-model for tension context: {e}")

        # 6. Intellectual Tension & Question Generation (MAX_CONCURRENT_LLM_REQUESTS = 2)
        sem_llm = asyncio.Semaphore(2)

        async def run_tension_for_fit(p):
            best_match = p.get("best_match")
            if not best_match:
                return p
                
            paper_title = best_match.get("paper_title", "")
            paper_abstract = next((pub["abstract"] for pub in p["publications"] if pub["title"] == paper_title), "")
            blocker_desc = best_match.get("blocker_description", "")
            inv_title = best_match.get("investigation_title", "")

            async with sem_llm:
                tension_res = await asyncio.to_thread(
                    self._llm.detect_intellectual_tension,
                    investigation_title=inv_title,
                    blocker_description=blocker_desc,
                    prof_paper_title=paper_title,
                    prof_paper_abstract=paper_abstract,
                    tension_threshold=0.70,
                    my_work_context=my_work_context
                )
            
            p["tension_score"] = tension_res.get("tension_score", 0.0)
            p["tension_analysis"] = tension_res.get("tension_description", "")
            if tension_res.get("has_tension", False):
                p["suggested_question"] = tension_res.get("intellectual_question", "")
                p["email_draft"] = tension_res.get("email_draft", "")
                
                # Persist versioned question to ProfessorMemory in SQLite database
                try:
                    prof_id = p["prof_id"]
                    prof_mem_dict = self._cache.get_professor_memory(prof_id)
                    if prof_mem_dict:
                        prof_mem = ProfessorMemory.from_dict(prof_mem_dict)
                    else:
                        prof_mem = ProfessorMemory.new(p["name"], inst_display_name, prof_id)
                    
                    versioned_question = {
                        "question": tension_res.get("intellectual_question", ""),
                        "email_draft": tension_res.get("email_draft", ""),
                        "self_context": tension_res.get("self_context", ""),
                        "paper_referenced": tension_res.get("paper_referenced") or paper_title,
                        "status": "generated",
                        "timestamp": time.time(),
                        "tension_score": tension_res.get("tension_score", 0.0),
                    }
                    
                    # Normalize existing questions
                    normalized = []
                    for q in prof_mem.questions_worth_asking:
                        if isinstance(q, str):
                            normalized.append({"question": q, "email_draft": "", "self_context": "", "paper_referenced": "", "status": "generated", "timestamp": 0.0})
                        else:
                            normalized.append(q)
                            
                    normalized.append(versioned_question)
                    prof_mem.questions_worth_asking = normalized
                    self._cache.save_professor_memory(prof_id, prof_mem.to_dict())
                except Exception as e:
                    logger.warning(f"Failed to save versioned question in university discovery for {p.get('name')}: {e}")
            else:
                p["suggested_question"] = None
                p["email_draft"] = None
            return p

        logger.info("Detecting intellectual tension concurrently for top programs...")
        tension_tasks = [run_tension_for_fit(p) for p in top_programs]
        scored_programs = await asyncio.gather(*tension_tasks)

        # 7. Generate Premium Markdown Report
        report_path = os.path.join(self._cache.cache_dir, f"{university_name.lower()}_discovery_report.md")
        self._write_markdown_report(report_path, university_name, inst_display_name, mode, active_invs, scored_programs)
        
        logger.info(f"University discovery completed. Report saved to: {report_path}")
        return report_path

    async def _extract_programs_for_dossier(self, dossier: dict, model_name: str) -> list[dict]:
        """
        Cluster professor publications into programs and summarize them using LLM.
        Also extracts first-class structured claims, methods, datasets, and metrics.
        """
        publications = dossier["publications"]
        if not publications:
            return []

        abstracts = [p.get("abstract", "").strip() for p in publications]
        valid_indices = [i for i, abs_text in enumerate(abstracts) if abs_text]

        # Concurrently extract claims/evidence for all papers with abstracts
        extract_tasks = []
        for idx in valid_indices:
            pub = publications[idx]
            extract_tasks.append(
                self._llm.extract_claims_and_evidence(pub.get("title", ""), pub.get("abstract", ""))
            )
        
        extracted_results = []
        if extract_tasks:
            extracted_results = await asyncio.gather(*extract_tasks)
            
        idx_to_extracted = {valid_indices[i]: res for i, res in enumerate(extracted_results)}

        # Helper to compile claims/metadata for a subset of publications
        def compile_program_claims(pub_indices: list[int]) -> dict:
            import uuid
            prog_claims = []
            prog_methods = set()
            prog_datasets = set()
            prog_metrics = set()
            prog_questions = []
            for idx in pub_indices:
                ext = idx_to_extracted.get(idx)
                if not ext:
                    continue
                doi = publications[idx].get("doi", "")
                for c_detail in ext.get("claims", []):
                    c_text = c_detail.get("claim", "")
                    if c_text:
                        prog_claims.append({
                            "id": str(uuid.uuid4()),
                            "claim": c_text,
                            "method": c_detail.get("method", "N/A"),
                            "dataset": c_detail.get("dataset", "N/A"),
                            "metric": c_detail.get("metric", "N/A"),
                            "confidence": c_detail.get("confidence", 1.0),
                            "paper_doi": doi
                        })
                for m in ext.get("methods", []):
                    if m and m.lower() != "n/a":
                        prog_methods.add(m)
                for d in ext.get("datasets", []):
                    if d and d.lower() != "n/a":
                        prog_datasets.add(d)
                for mt in ext.get("metrics", []):
                    if mt and mt.lower() != "n/a":
                        prog_metrics.add(mt)
                prog_questions.extend(ext.get("open_questions", []))
            return {
                "claims": prog_claims,
                "methods": list(prog_methods),
                "datasets": list(prog_datasets),
                "metrics": list(prog_metrics),
                "open_questions": prog_questions
            }

        # Fallback if fewer than 2 valid papers with abstracts
        if len(valid_indices) < 2:
            key_terms = self._extract_cluster_keywords(publications)
            fallback_title = " & ".join(t.capitalize() for t in key_terms[:3]) or "Exploratory Research"
            meta = compile_program_claims(valid_indices)
            return [{
                "title": fallback_title,
                "summary": f"Research program focusing on {', '.join(key_terms[:5])}.",
                "publications": publications,
                "claims": meta["claims"],
                "methods": meta["methods"],
                "datasets": meta["datasets"],
                "metrics": meta["metrics"],
                "open_questions": meta["open_questions"]
            }]

        valid_abstracts = [abstracts[i] for i in valid_indices]
        
        # Get embeddings via cached_encode
        embeddings = await asyncio.to_thread(
            cached_encode, valid_abstracts, model_name="BAAI/bge-large-en-v1.5", normalize_embeddings=True
        )

        labels = self._cluster_papers(publications, embeddings)

        clustered_groups = {}
        for idx, label in enumerate(labels):
            cluster_id = int(label)
            real_idx = valid_indices[idx]
            if cluster_id not in clustered_groups:
                clustered_groups[cluster_id] = []
            clustered_groups[cluster_id].append(real_idx)

        noise_papers_indices = []
        for idx in range(len(publications)):
            if idx not in valid_indices:
                noise_papers_indices.append(idx)
        if -1 in clustered_groups:
            noise_papers_indices.extend(clustered_groups[-1])

        programs = []
        summarize_tasks = []
        cluster_ids = [cid for cid in clustered_groups.keys() if cid >= 0]

        for cid in cluster_ids:
            indices_in_cluster = clustered_groups[cid]
            papers_in_cluster = [publications[idx] for idx in indices_in_cluster]
            key_terms = self._extract_cluster_keywords(papers_in_cluster)
            task = self._llm.summarize_research_program(papers_in_cluster, key_terms)
            summarize_tasks.append((indices_in_cluster, papers_in_cluster, key_terms, task))

        if summarize_tasks:
            results = await asyncio.gather(*(t[3] for t in summarize_tasks))
            for (indices_in_cluster, papers_in_cluster, key_terms, _), res in zip(summarize_tasks, results):
                meta = compile_program_claims(indices_in_cluster)
                programs.append({
                    "title": res.get("title", "Exploratory Research"),
                    "summary": res.get("summary", ""),
                    "publications": papers_in_cluster,
                    "claims": meta["claims"],
                    "methods": meta["methods"],
                    "datasets": meta["datasets"],
                    "metrics": meta["metrics"],
                    "open_questions": meta["open_questions"]
                })

        if noise_papers_indices:
            noise_papers = [publications[idx] for idx in noise_papers_indices]
            key_terms = self._extract_cluster_keywords(noise_papers)
            meta = compile_program_claims([idx for idx in noise_papers_indices if idx in valid_indices])
            programs.append({
                "title": "General & Exploratory Research",
                "summary": f"General research topics covering {', '.join(key_terms[:5])}.",
                "publications": noise_papers,
                "claims": meta["claims"],
                "methods": meta["methods"],
                "datasets": meta["datasets"],
                "metrics": meta["metrics"],
                "open_questions": meta["open_questions"]
            })

        return programs

    def _cluster_papers(self, papers: list[dict], paper_embeddings: np.ndarray) -> np.ndarray:
        """
        Cluster papers using HDBSCAN, falling back to AgglomerativeClustering.
        """
        N = len(papers)
        if N < 2:
            return np.zeros(N, dtype=int)

        try:
            from sklearn.cluster import HDBSCAN
            clusterer = HDBSCAN(min_cluster_size=2, min_samples=1, metric='euclidean')
            labels = clusterer.fit_predict(paper_embeddings)
            
            unique_labels = set(labels)
            non_noise_clusters = [l for l in unique_labels if l >= 0]
            
            if len(non_noise_clusters) >= 2:
                logger.info(f"HDBSCAN discovered {len(non_noise_clusters)} research programs.")
                return labels
                
            logger.info("HDBSCAN yielded fewer than 2 research programs. Falling back to AgglomerativeClustering...")
        except Exception as e:
            logger.warning(f"HDBSCAN clustering failed: {e}. Falling back to AgglomerativeClustering...")

        # Fallback Agglomerative
        try:
            from sklearn.cluster import AgglomerativeClustering
            n_clusters = min(3, N)
            if n_clusters >= 2:
                clusterer = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
                labels = clusterer.fit_predict(paper_embeddings)
                logger.info(f"AgglomerativeClustering structured {n_clusters} research programs.")
                return labels
        except Exception as e:
            logger.warning(f"AgglomerativeClustering fallback failed: {e}. Treating all papers as one program.")

        return np.zeros(N, dtype=int)

    def _extract_cluster_keywords(self, papers: list[dict]) -> list[str]:
        """
        Extract top TF-IDF keywords from a cluster of papers.
        """
        import re
        from collections import Counter
        import math

        STOPWORDS = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "by", "from", "this", "that", "these",
            "those", "is", "are", "was", "were", "be", "been", "have",
            "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "we", "our", "us", "it", "its",
            "their", "they", "as", "which", "such", "not", "also", "both",
            "can", "into", "than", "through", "using", "based", "show",
            "shows", "propose", "result", "results", "paper", "work",
            "method", "approach", "framework", "system", "model", "algorithm",
            "performance", "analysis", "evaluation", "experiments", "study",
            "explore", "explores", "exploring", "exploration", "proposes",
            "proposed", "proposing", "methods", "approaches", "frameworks",
            "systems", "models", "algorithms", "studies", "analyze", "analyzes",
            "analyzed", "analyzing", "investigate", "investigates", "investigated",
            "investigating", "provide", "provides", "provided", "providing",
            "present", "presents", "presented", "presenting", "demonstrate",
            "demonstrates", "demonstrated", "demonstrating", "evaluate",
            "evaluates", "evaluated", "evaluating", "compare", "compares",
            "compared", "comparing"
        }

        texts = [
            (p.get("title", "") + " " + (p.get("abstract") or p.get("abstract_snippet", ""))).lower()
            for p in papers
        ]

        all_words = []
        doc_word_sets = []
        for text in texts:
            doc_words = set()
            for word in re.findall(r"\b[a-zA-Z]{4,}\b", text):
                if word not in STOPWORDS:
                    all_words.append(word)
                    doc_words.add(word)
            doc_word_sets.append(doc_words)

        if not all_words:
            return []

        tf = Counter(all_words)
        n_docs = max(len(doc_word_sets), 1)

        df: dict[str, int] = Counter()
        for dws in doc_word_sets:
            for w in dws:
                df[w] += 1

        scored = {}
        for word, freq in tf.items():
            density = df.get(word, 0) / n_docs
            scored[word] = freq * (1.0 + density)

        sorted_words = [w for w, _ in sorted(scored.items(), key=lambda x: -x[1])]
        return sorted_words[:5]

    def _write_markdown_report(
        self,
        filepath: str,
        university_query: str,
        university_name: str,
        mode: str,
        active_invs: list,
        profiles: list[dict]
    ):
        """Generates the premium Markdown report."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        lines = [
            f"# Research Network OS — Discovery Report: {university_name}",
            f"**Generated At**: {timestamp} | **Search Query**: `{university_query}`",
            f"**Discovery Mode**: Mode {'B (Internship Targets)' if mode == 'internship' else 'A (Best Researchers)'}",
            "",
            "## Active Investigations In Context",
            "This discovery run was prioritized based on the active blockers of the following investigations:",
            ""
        ]
        
        if active_invs:
            for inv in active_invs:
                lines.append(f"- **{inv.title}** (Priority: `{inv.effective_priority:.2f}`)")
                for b in inv.open_blockers:
                    lines.append(f"  - *Blocker*: {b.description} (`{b.blocker_type}` | Severity: `{b.severity:.2f}`)")
        else:
            lines.append("- *None active. Defaults applied across core Machine Learning, AI, Cognitive Science, and Neuroscience concepts.*")

        lines.extend([
            "",
            "---",
            "",
            "## Executive Rankings",
            "Below is the list of top ranked research programs matching your goals. Click the program name to view detailed alignment profile.",
            "",
            "| Rank | Research Program | Professor | Research Utility | Accessibility | Recommended Timing | Primary Action |",
            "|------|------------------|-----------|------------------|---------------|--------------------|----------------|"
        ])
        
        for idx, p in enumerate(profiles):
            prog_anchor = f"{p['program_title'].lower().replace(' ', '-').replace('&', '')}-{p['name'].lower().replace(' ', '-')}"
            lines.append(
                f"| {idx+1} | [{p['program_title']}](#{prog_anchor}) | {p['name']} | **{p['research_utility']:.1f}/100** | {p['accessibility_score']:.1f}/100 | **{p['contact_timing']}** | {p['recommended_action']} |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## Deep Program Profiles & Question Drafts",
            ""
        ])

        for idx, p in enumerate(profiles):
            best_paper_title = "None"
            best_blocker_desc = "None"
            if p.get("best_match"):
                best_paper_title = p["best_match"].get("paper_title", "None")
                best_blocker_desc = p["best_match"].get("blocker_description", "None")

            prog_anchor = f"{p['program_title'].lower().replace(' ', '-').replace('&', '')}-{p['name'].lower().replace(' ', '-')}"
            lines.extend([
                f"### {idx+1}. {p['program_title']} — {p['name']}",
                f"- **Affiliation**: {p['research_areas'][0] if p.get('research_areas') else 'Computer Science'}, {p['position']} @ {university_name}",
                f"- **PI H-Index**: {p['h_index']} | **PI Last Pub Year**: {p['last_pub_year']} | **PI Supervision Signal**: {p['student_supervision_signal']:.2f}",
                "",
                "#### Research Program Summary",
                f"*{p['program_summary']}*",
                "",
                "#### Research Program Graph Map",
                "List of asserted claims and supporting methodology:"
            ])

            claims = p.get("claims", [])
            if claims:
                for c in claims:
                    lines.append(f"- **Claim**: *\"{c['claim']}\"* (Confidence: `{c.get('confidence', 1.0):.2f}`)")
                    lines.append(f"  - **Method**: `{c.get('method', 'N/A')}` | **Dataset**: `{c.get('dataset', 'N/A')}` | **Metric**: `{c.get('metric', 'N/A')}`")
            else:
                lines.append("- *No structured claims extracted for this program.*")
            lines.append("")

            lines.append("#### Direct Contradictions & Tensions")
            contradictions = p.get("contradictions", [])
            if contradictions:
                for c in contradictions:
                    lines.extend([
                        "> [!WARNING]",
                        f"> **Direct Contradiction (Severity: {c['severity']:.2f}, Confidence: {c['confidence']:.2f})**",
                        f"> - *Your Claim*: \"{c['my_claim']}\"",
                        f"> - *Their Claim*: \"{c['their_claim']}\"",
                        f"> - *Resolution Hypothesis*: {c['resolution_hypothesis']}",
                        ""
                    ])
            else:
                lines.append("No direct contradictions detected with your research claims.")
                lines.append("")

            lines.extend([
                "#### Fit Summary",
                f"- **Research Utility**: **`{p['research_utility']:.1f}/100`**",
                f"- **Internship Accessibility**: **`{p['accessibility_score']:.1f}/100`**",
                f"- **Top Relevant Paper in Program**: *{best_paper_title}*",
                f"- **Most Relevant Blocker**: *{best_blocker_desc}*",
                f"- **Recommended First Question**: *{p.get('suggested_question') or 'None'}*",
                f"- **Recommended Contact Timing**: **`{p['contact_timing']}`**",
                "",
                "#### Paper-Level Research Overlaps",
            ])
            
            if p.get("paper_overlaps"):
                for item in p["paper_overlaps"][:3]:  # Display top 3 overlaps
                    themes_str = "\n".join(f"  - {theme}" for theme in item["shared_themes"]) if item.get("shared_themes") else "  - general topical similarity"
                    lines.extend([
                        "**YOUR PAPER**",
                        "-----------",
                        f"{item['user_paper_title']}",
                        "",
                        "**THEIR PAPER**",
                        "-----------",
                        f"{item['prof_paper_title']}",
                        "",
                        f"**Similarity**: `{item['similarity']:.4f}`",
                        "**Shared Themes**:",
                        themes_str,
                        ""
                    ])
            else:
                lines.append("No significant paper-level research overlaps found.")
                lines.append("")

            lines.extend([
                "#### Dimension Breakdown",
                "```text"
            ])
            
            dims = p.get("dimensions", {})
            DIM_LABELS = {
                "enrs":           "ENRS             ",
                "topic_overlap":  "Topic Overlap    ",
                "gap_complement": "Gap Complement   ",
                "method_compat":  "Method Compat    ",
                "lab_access":     "Lab Access       ",
                "temporal_act":   "Temporal Act.    ",
            }
            
            for key, label in DIM_LABELS.items():
                if key not in dims:
                    continue
                d = dims[key]
                raw = d.get("raw", 0.0)
                contrib = d.get("contribution", 0.0)
                pct = d.get("weight", 0.0) * 100
                bar_filled = int(raw * 20)
                bar = "█" * bar_filled + "░" * (20 - bar_filled)
                lines.append(f"  {label} {bar}  {contrib:5.1f} pts  ({pct:.0f}%)")
                
            lines.extend([
                "```",
                ""
            ])

            match = p.get("best_match")
            if match:
                w_dict = match.get("weights") or {}
                w_sem = w_dict.get("semantic", 0.30)
                w_meth = w_dict.get("method", 0.25)
                w_data = w_dict.get("dataset", 0.0)
                w_meas = w_dict.get("measurement", 0.10)
                lines.extend([
                    "#### Blocker Resolution Analysis",
                    "> [!NOTE]",
                    f"> **Matched Blocker**: *{match.get('blocker_description')}*",
                    f"> **Supporting Paper**: [{match.get('paper_title')}]({match.get('paper_doi')}) ({match.get('paper_year')})",
                    f"> **Blocker Resolution Confidence**: **`{match.get('confidence', 0.0):.4f}`** (Cosine Similarity: `{match.get('similarity'):.4f}`)",
                    "> ",
                    "> | Factor | Score | Weight | Contribution |",
                    "> |--------|-------|--------|--------------|",
                    f"> | Semantic | {match.get('semantic_factor', 0.0):.4f} | {w_sem:.2f} | {match.get('semantic_factor', 0.0) * w_sem:.4f} |",
                    f"> | Method | {match.get('method_factor', 0.0):.4f} | {w_meth:.2f} | {match.get('method_factor', 0.0) * w_meth:.4f} |",
                    f"> | Dataset | {match.get('dataset_factor', 0.0):.4f} | {w_data:.2f} | {match.get('dataset_factor', 0.0) * w_data:.4f} |",
                    f"> | Measurement | {match.get('measurement_factor', 0.0):.4f} | {w_meas:.2f} | {match.get('measurement_factor', 0.0) * w_meas:.4f} |",
                    ""
                ])
                
                lines.append("#### Intellectual Tension Draft")
                if p.get("suggested_question"):
                    lines.extend([
                        f"**Tension Score**: `{p['tension_score']:.2f}` (Passed 0.70 Gate)",
                        f"*{p['tension_analysis']}*",
                        "",
                        "**Suggested Question**:",
                        "```text",
                        p["suggested_question"],
                        "```",
                        "",
                        "**Drafted Outreach Email**:",
                        "```text",
                        p.get("email_draft") or "No email draft generated.",
                        "```"
                    ])
                else:
                    lines.extend([
                        f"**Tension Score**: `{p.get('tension_score', 0.0):.2f}` (Below 0.70 Gate)",
                        "Suggested question omitted. The intellectual overlap is insufficient to formulate a high-credibility peer question.",
                        ""
                    ])
            else:
                lines.extend([
                    "#### Blocker Resolution Analysis",
                    "No direct blocker match above the required semantic similarity threshold. Overlap is general topical similarity.",
                    ""
                ])

            lines.extend([
                "#### Key Program Publications",
                ""
            ])
            for pub_idx, pub in enumerate(p.get("publications", [])):
                venue_str = f" | **Venue**: *{pub['venue']}*" if pub.get("venue") else ""
                cites_str = f" | **Citations**: {pub['citations']}" if pub.get("citations") is not None else ""
                lines.extend([
                    f"{pub_idx+1}. **{pub['title']}** ({pub['year']}){venue_str}{cites_str}",
                    f"   - **DOI/URL**: [{pub['doi']}]({pub['doi']})"
                ])
                abstract_snippet = pub.get("abstract", "").strip()
                if abstract_snippet:
                    snippet = abstract_snippet if len(abstract_snippet) <= 200 else abstract_snippet[:200] + "..."
                    lines.append(f"   - **Abstract Snippet**: *{snippet}*")
                lines.append("")

            lines.extend([
                "#### Recommended Next Action",
                f"- **{p['recommended_action']}**",
                "",
                "---",
                ""
            ])

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
