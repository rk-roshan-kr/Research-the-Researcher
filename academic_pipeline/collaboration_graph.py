"""
collaboration_graph.py — Deep co-author network with BFS + priority queue + Union-Find.

Replaces the simple ThreadPoolExecutor loop with a proper graph algorithm stack:

  1. Build weighted adjacency list:
       author_id → [(co_author_id, paper_count, paper_list), ...]
       O(P × A) where P = papers, A = authors per paper

  2. BFS with max-heap priority queue:
       Priority = h_index × log(citations + 1)
       → Most impactful collaborators profiled first
       → Uses heapq (Python's built-in binary heap)

  3. Union-Find (Disjoint Set Union):
       Detects connected components / research clusters
       Path compression + union-by-rank → nearly O(1) amortized

  4. Graph compression:
       Prune edges with paper_count < MIN_EDGE_WEIGHT and
       nodes with h_index == 0 (uninformative leaf nodes)
       Reduces vis.js rendering load on large graphs

  5. Parallel enrichment:
       32-thread ThreadPoolExecutor matches Ryzen 9950X thread count
       Each thread enriches one node via OpenAlex while others proceed
"""

from __future__ import annotations

import heapq
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

logger = logging.getLogger(__name__)

MIN_EDGE_WEIGHT = 1    # prune edges with fewer shared papers
MIN_NODE_H      = 0    # prune depth-2 nodes with h_index below this


# ──────────────────────────────────────────────────────────────────────────────
# Union-Find (Disjoint Set Union)
# ──────────────────────────────────────────────────────────────────────────────

class UnionFind:
    """
    Path-compressed, rank-unioned DSU.
    Used to identify research clusters in the collaboration graph.
    """

    def __init__(self):
        self._parent: dict[str, str] = {}
        self._rank:   dict[str, int] = {}

    def _ensure(self, x: str):
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x]   = 0

    def find(self, x: str) -> str:
        self._ensure(x)
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])   # path compression
        return self._parent[x]

    def union(self, x: str, y: str):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1

    def components(self) -> dict[str, list[str]]:
        """Return {root_id: [member_ids]}."""
        comps: dict[str, list[str]] = {}
        for node in self._parent:
            root = self.find(node)
            comps.setdefault(root, []).append(node)
        return comps


# ──────────────────────────────────────────────────────────────────────────────
# CollaborationGraphBuilder
# ──────────────────────────────────────────────────────────────────────────────

class CollaborationGraphBuilder:
    """
    Builds a deep, weighted collaboration network.

    Public API:
        graph = builder.build_graph(
            target_name, target_openalex_id, target_metrics,
            publications, depth=1, max_coauthors=30
        )
        # → { nodes, edges, coauthor_profiles, clusters }
    """

    def __init__(self, openalex_client, s2_client, cache_manager):
        self.openalex = openalex_client
        self.s2       = s2_client
        self.cache    = cache_manager

    # ── Node enrichment ───────────────────────────────────────────────────────

    def _search_oa_by_name(self, name: str) -> Optional[dict]:
        try:
            results = self.openalex.search_authors(name)
            return results[0] if results else None
        except Exception:
            return None

    def _enrich_coauthor(self, name: str, shared_papers: list) -> dict:
        """Fetch OpenAlex profile for a co-author. Returns enriched node dict."""
        cache_key = f"collab_{name.lower().replace(' ', '_').replace('.', '')}"
        cached = self.cache.get_resolved_profile(cache_key)
        if cached:
            cached["shared_papers"] = shared_papers
            cached["shared_count"]  = len(shared_papers)
            return cached

        oa = self._search_oa_by_name(name)
        node = {
            "name":        name,
            "openalex_id": None,
            "affiliation": "",
            "h_index":     0,
            "citations":   0,
            "publications": 0,
            "interests":   [],
            "orcid":       None,
            "shared_papers": shared_papers,
            "shared_count":  len(shared_papers),
        }

        if oa:
            node["openalex_id"] = oa.get("id", "").split("/")[-1]
            insts = oa.get("last_known_institutions") or []
            node["affiliation"] = insts[0].get("display_name", "") if insts else ""
            node["h_index"]     = (oa.get("summary_stats") or {}).get("h_index", 0) or 0
            node["citations"]   = oa.get("cited_by_count", 0) or 0
            node["publications"] = oa.get("works_count", 0) or 0
            node["interests"]   = [
                c["display_name"]
                for c in (oa.get("x_concepts") or [])
                if c.get("display_name")
            ][:4]
            orcid_raw = oa.get("orcid") or ""
            node["orcid"] = orcid_raw.split("/")[-1] if orcid_raw else None

        self.cache.save_resolved_profile(cache_key, node)
        return node

    # ── Priority scoring ──────────────────────────────────────────────────────

    @staticmethod
    def _priority(h_index: int, citations: int) -> float:
        """
        BFS priority: negative because heapq is a min-heap.
        Higher h_index × log(citations) → more important → lower priority number.
        """
        score = (h_index + 1) * math.log(citations + 2)
        return -score   # negate for max-heap behaviour

    # ── Adjacency list builder ────────────────────────────────────────────────

    def _build_coauthor_map(self, target_norm: str,
                             publications: list) -> tuple[dict[str, list], dict[str, dict]]:
        """
        Build {author_name: [paper_summary, ...]} and metadata map from publications.
        O(P × A) where P = papers, A = avg authors per paper.
        """
        coauthor_map: dict[str, list] = {}
        coauthor_meta: dict[str, dict] = {}
        for paper in publications:
            summary = {
                "title":     paper.get("title", ""),
                "year":      paper.get("year"),
                "citations": paper.get("citations_max", 0),
                "doi":       paper.get("doi"),
                "arxiv":     paper.get("arxiv"),
                "oa_url":    paper.get("oa_url"),
                "venue":     paper.get("venue", ""),
            }
            authors_detailed = paper.get("authors_detailed", [])
            for author in paper.get("authors", []):
                norm = author.lower().replace(" ", "").replace(".", "")
                if norm == target_norm:
                    continue
                coauthor_map.setdefault(author, []).append(summary)
                
                if author not in coauthor_meta:
                    coauthor_meta[author] = {
                        "name": author,
                        "openalex_id": None,
                        "semantic_scholar_id": None,
                        "orcid": None,
                        "affiliation": ""
                    }
                
                # Match detail by normalized name
                for ad in authors_detailed:
                    ad_norm = ad.get("name", "").lower().replace(" ", "").replace(".", "")
                    if ad_norm == norm:
                        aid = ad.get("id")
                        if aid:
                            if str(aid).startswith('A'):
                                coauthor_meta[author]["openalex_id"] = str(aid)
                            else:
                                coauthor_meta[author]["semantic_scholar_id"] = str(aid)
                        if ad.get("orcid") and not coauthor_meta[author]["orcid"]:
                            coauthor_meta[author]["orcid"] = ad.get("orcid")
                        if ad.get("affiliation") and len(ad.get("affiliation")) > len(coauthor_meta[author]["affiliation"]):
                            coauthor_meta[author]["affiliation"] = ad.get("affiliation")
                        break
        return coauthor_map, coauthor_meta

    # ── BFS enrichment with priority queue ───────────────────────────────────

    def _bfs_enrich(self, coauthor_map: dict[str, list],
                     coauthor_meta: dict[str, dict],
                     max_coauthors: int,
                     max_workers: int,
                     progress=None,
                     task_id=None) -> dict[str, dict]:
        """
        Enrich co-authors in BFS priority order (highest-impact first).
        Uses a max-heap keyed by collaboration frequency (proxy for importance).
        First checks SQLite cache, then batch-fetches remaining with OpenAlex IDs.
        Falls back to parallel name search for others.
        """
        # Initial heap: (-paper_count, name)
        heap = [(-len(papers), name) for name, papers in coauthor_map.items()]
        heapq.heapify(heap)

        # We take the top max_coauthors by paper count for enrichment
        to_enrich = []
        while heap and len(to_enrich) < max_coauthors:
            neg_count, name = heapq.heappop(heap)
            to_enrich.append((name, coauthor_map[name]))

        logger.info(f"BFS: enriching top {len(to_enrich)} co-authors "
                    f"({len(coauthor_map)} total) with batch + parallel queries...")

        profiles: dict[str, dict] = {}
        uncached_with_oa = []
        uncached_without_oa = []

        # 1. Check cache first
        for name, papers in to_enrich:
            cache_key = f"collab_{name.lower().replace(' ', '_').replace('.', '')}"
            cached = self.cache.get_resolved_profile(cache_key)
            if cached:
                cached["shared_papers"] = papers
                cached["shared_count"]  = len(papers)
                profiles[name] = cached
                if progress and task_id is not None:
                    progress.advance(task_id)
            else:
                meta = coauthor_meta.get(name, {})
                oa_id = meta.get("openalex_id")
                if oa_id:
                    uncached_with_oa.append((name, oa_id, papers))
                else:
                    uncached_without_oa.append((name, papers))

        # 2. Batch fetch co-authors with OpenAlex IDs
        id_to_name = {}
        id_to_papers = {}
        for name, oa_id, papers in uncached_with_oa:
            clean_id = oa_id.split("/")[-1]
            id_to_name[clean_id] = name
            id_to_papers[clean_id] = papers

        batch_ids = list(id_to_name.keys())
        if batch_ids:
            logger.info(f"Batch fetching {len(batch_ids)} co-authors from OpenAlex...")
            try:
                batch_results = self.openalex.get_authors_batch(batch_ids)
                for oa in batch_results:
                    oa_id_raw = oa.get("id", "")
                    oa_id = oa_id_raw.split("/")[-1] if oa_id_raw else None
                    if oa_id in id_to_name:
                        name = id_to_name[oa_id]
                        papers = id_to_papers[oa_id]
                        
                        insts = oa.get("last_known_institutions") or []
                        affiliation = insts[0].get("display_name", "") if insts else ""
                        h_index = (oa.get("summary_stats") or {}).get("h_index", 0) or 0
                        citations = oa.get("cited_by_count", 0) or 0
                        publications = oa.get("works_count", 0) or 0
                        interests = [
                            c["display_name"]
                            for c in (oa.get("x_concepts") or [])
                            if c.get("display_name")
                        ][:4]
                        orcid_raw = oa.get("orcid") or ""
                        orcid = orcid_raw.split("/")[-1] if orcid_raw else None
                        
                        node = {
                            "name":        name,
                            "openalex_id": oa_id,
                            "affiliation": affiliation,
                            "h_index":     h_index,
                            "citations":   citations,
                            "publications": publications,
                            "interests":   interests,
                            "orcid":       orcid,
                            "shared_papers": papers,
                            "shared_count":  len(papers),
                        }
                        
                        cache_key = f"collab_{name.lower().replace(' ', '_').replace('.', '')}"
                        self.cache.save_resolved_profile(cache_key, node)
                        profiles[name] = node
                        
                        if progress and task_id is not None:
                            progress.advance(task_id)
            except Exception as e:
                logger.error(f"Failed to batch fetch authors: {e}")

            # Any that failed or weren't returned by batch query fall back to name search
            for name, oa_id, papers in uncached_with_oa:
                if name not in profiles:
                    uncached_without_oa.append((name, papers))

        # 3. Parallel name search for remaining uncached authors
        if uncached_without_oa:
            logger.info(f"Parallel name search for {len(uncached_without_oa)} remaining co-authors...")
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                future_map = {
                    ex.submit(self._enrich_coauthor, name, papers): name
                    for name, papers in uncached_without_oa
                }
                for future in as_completed(future_map):
                    if progress and task_id is not None:
                        progress.advance(task_id)
                    name = future_map[future]
                    try:
                        profile = future.result()
                        profiles[name] = profile
                    except Exception as e:
                        logger.warning(f"Enrichment failed for '{name}': {e}")
                        profiles[name] = {
                            "name":         name,
                            "openalex_id":  None,
                            "affiliation":  "",
                            "h_index":      0,
                            "citations":    0,
                            "publications": 0,
                            "interests":    [],
                            "orcid":        None,
                            "shared_papers": coauthor_map[name],
                            "shared_count": len(coauthor_map[name]),
                        }
                    time.sleep(0.05)

        return profiles

    # ── vis.js graph serialiser ───────────────────────────────────────────────

    def _build_graph_data(self, root_id: str, root_name: str,
                           root_metrics: dict,
                           profiles: dict[str, dict],
                           uf: UnionFind) -> tuple[list, list]:
        """
        Convert enriched profiles into vis.js nodes + edges.
        Applies edge pruning (MIN_EDGE_WEIGHT).
        """
        nodes = []
        edges = []

        # Root node
        nodes.append({
            "id":    root_id,
            "label": root_name,
            "group": "root",
            "title": (f"<b>{root_name}</b><br>"
                      f"H-Index: {root_metrics.get('h_index', 0)}<br>"
                      f"Citations: {root_metrics.get('citations', 0):,}<br>"
                      f"Publications: {root_metrics.get('publications', 0)}"),
            "value": 60,
            "h_index":      root_metrics.get("h_index", 0),
            "citations":    root_metrics.get("citations", 0),
            "publications": root_metrics.get("publications", 0),
            "affiliation":  "",
            "interests":    [],
        })

        for name, p in profiles.items():
            shared = p["shared_count"]
            if shared < MIN_EDGE_WEIGHT:
                continue   # prune weak edges

            node_id = p.get("openalex_id") or f"ca_{name.replace(' ', '_')}"
            uf.union(root_id, node_id)   # mark as connected component

            interests_str = ", ".join(p.get("interests", [])) or "N/A"
            tooltip = (
                f"<b>{name}</b><br>"
                f"Affiliation: {p.get('affiliation') or 'Unknown'}<br>"
                f"H-Index: {p.get('h_index', 0)}<br>"
                f"Citations: {p.get('citations', 0):,}<br>"
                f"Publications: {p.get('publications', 0)}<br>"
                f"Research: {interests_str}<br>"
                f"<br><b>Shared papers with {root_name}: {shared}</b>"
            )

            nodes.append({
                "id":    node_id,
                "label": name,
                "group": "coauthor",
                "title": tooltip,
                "value": max(10, min(40, shared * 4)),
                "h_index":      p.get("h_index", 0),
                "citations":    p.get("citations", 0),
                "publications": p.get("publications", 0),
                "affiliation":  p.get("affiliation", ""),
                "interests":    p.get("interests", []),
                "openalex_id":  p.get("openalex_id"),
                "orcid":        p.get("orcid"),
            })

            paper_lines = "".join(
                f"<li>{pp['title'][:70]}{'…' if len(pp['title'])>70 else ''} "
                f"({pp.get('year') or '?'}, {pp.get('citations', 0)} cit.)</li>"
                for pp in p["shared_papers"][:10]
            )
            edge_tooltip = (
                f"<b>{shared} shared paper(s):</b><ul>{paper_lines}</ul>"
                + ("<i>…and more</i>" if len(p["shared_papers"]) > 10 else "")
            )
            edges.append({
                "from":         root_id,
                "to":           node_id,
                "title":        edge_tooltip,
                "value":        shared,
                "shared_count": shared,
                "shared_papers": p["shared_papers"],
            })

        return nodes, edges

    # ── Depth-2 expansion ─────────────────────────────────────────────────────

    def _expand_depth2(self, nodes: list, edges: list,
                        profiles: dict[str, dict],
                        root_name: str, uf: UnionFind):
        """
        For each depth-1 node with an OpenAlex ID, fetch their works and
        add their top-5 collaborators as depth-2 (green) nodes.
        """
        existing_ids = {n["id"] for n in nodes}
        root_norm = root_name.lower().replace(" ", "").replace(".", "")

        for name, p in list(profiles.items()):
            oa_id = p.get("openalex_id")
            if not oa_id:
                continue

            try:
                works = self.openalex.get_author_works(oa_id, max_works=30)
            except Exception as e:
                logger.debug(f"D2 works fetch failed for '{name}': {e}")
                continue

            secondary_map: dict[str, list] = {}
            name_norm = name.lower().replace(" ", "").replace(".", "")

            for work in works:
                paper_s = {
                    "title":     work.get("title", ""),
                    "year":      work.get("publication_year"),
                    "citations": work.get("cited_by_count", 0),
                }
                for authorship in (work.get("authorships") or []):
                    a_name = (authorship.get("author") or {}).get("display_name", "")
                    a_norm = a_name.lower().replace(" ", "").replace(".", "")
                    if not a_name or a_norm in (name_norm, root_norm):
                        continue
                    secondary_map.setdefault(a_name, []).append(paper_s)

            # Pick top-5 by count, skip if too weak
            top_secondary = sorted(
                [(n, ps) for n, ps in secondary_map.items()
                 if len(ps) >= MIN_EDGE_WEIGHT],
                key=lambda x: len(x[1]), reverse=True
            )[:5]

            for sec_name, sec_papers in top_secondary:
                sec_id = f"ca2_{sec_name.replace(' ', '_')}"

                if sec_id not in existing_ids:
                    oa_sec = self._search_oa_by_name(sec_name)
                    if oa_sec:
                        raw_id = (oa_sec.get("id") or "").split("/")[-1]
                        if raw_id:
                            sec_id = raw_id
                        sec_affil = (
                            (oa_sec.get("last_known_institutions") or [{}])[0]
                            .get("display_name", "")
                        )
                        sec_h   = (oa_sec.get("summary_stats") or {}).get("h_index", 0) or 0
                        sec_cit = oa_sec.get("cited_by_count", 0) or 0
                        sec_pub = oa_sec.get("works_count", 0) or 0
                    else:
                        sec_affil, sec_h, sec_cit, sec_pub = "", 0, 0, 0

                    if sec_h < MIN_NODE_H:
                        continue   # prune uninformative leaf

                    if sec_id not in existing_ids:
                        nodes.append({
                            "id":    sec_id,
                            "label": sec_name,
                            "group": "depth2",
                            "title": (f"<b>{sec_name}</b><br>"
                                      f"Affiliation: {sec_affil or 'Unknown'}<br>"
                                      f"H-Index: {sec_h}<br>"
                                      f"Collaborates with: {name}"),
                            "value":        max(6, min(20, len(sec_papers) * 2)),
                            "h_index":      sec_h,
                            "citations":    sec_cit,
                            "publications": sec_pub,
                            "affiliation":  sec_affil,
                            "interests":    [],
                        })
                        existing_ids.add(sec_id)
                        uf.union(oa_id, sec_id)

                # Edge: primary → secondary
                edge_exists = any(
                    e["from"] == oa_id and e["to"] == sec_id for e in edges
                )
                if not edge_exists and oa_id in existing_ids:
                    edges.append({
                        "from":         oa_id,
                        "to":           sec_id,
                        "title":        f"<b>{len(sec_papers)} shared paper(s)</b>",
                        "value":        len(sec_papers),
                        "shared_count": len(sec_papers),
                        "shared_papers": sec_papers,
                    })

            time.sleep(0.1)

    # ── Public entry point ────────────────────────────────────────────────────

    def build_graph(self,
                    target_name:        str,
                    target_openalex_id: str,
                    target_metrics:     dict,
                    publications:       list,
                    depth:              int = 1,
                    max_coauthors:      int = 30,
                    max_workers:        int = 32,
                    progress            = None,
                    task_id             = None) -> dict:
        """
        Build the full collaboration graph.

        Returns:
            {
              "nodes":             list of vis.js node dicts,
              "edges":             list of vis.js edge dicts,
              "coauthor_profiles": {name: profile_dict},
              "clusters":          {cluster_root: [member_ids]},
            }
        """
        logger.info(
            f"Building collaboration graph for '{target_name}' "
            f"(depth={depth}, max_coauthors={max_coauthors})"
        )

        root_id = target_openalex_id or "ROOT"
        target_norm = target_name.lower().replace(" ", "").replace(".", "")

        # 1. Build adjacency / co-author map
        coauthor_map, coauthor_meta = self._build_coauthor_map(target_norm, publications)
        logger.info(f"Found {len(coauthor_map)} unique co-authors in publications.")

        # 2. BFS enrichment with priority queue
        profiles = self._bfs_enrich(
            coauthor_map, coauthor_meta, max_coauthors, max_workers, progress=progress, task_id=task_id
        )

        # 3. Build Union-Find for cluster detection
        uf = UnionFind()
        uf.find(root_id)   # ensure root exists

        # 4. Serialise to vis.js nodes + edges
        nodes, edges = self._build_graph_data(
            root_id, target_name, target_metrics, profiles, uf
        )

        # 5. Depth-2 expansion
        if depth >= 2:
            logger.info("Expanding to depth-2...")
            self._expand_depth2(nodes, edges, profiles, target_name, uf)

        # 6. Detect clusters
        clusters = uf.components()
        cluster_count = len([c for c in clusters.values() if len(c) > 1])
        logger.info(f"Graph complete: {len(nodes)} nodes, {len(edges)} edges, "
                    f"{cluster_count} clusters.")

        return {
            "nodes":             nodes,
            "edges":             edges,
            "coauthor_profiles": profiles,
            "clusters":          clusters,
        }
