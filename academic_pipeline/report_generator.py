import os
import logging
from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)

class ReportGenerator:
    """
    Renders structured academic profile dossiers into beautiful Markdown and interactive HTML reports.
    """
    def __init__(self, template_dir=None):
        if template_dir is None:
            # Look inside the package package templates directory
            template_dir = os.path.join(os.path.dirname(__file__), "templates")
        self.template_dir = template_dir
        
        # Initialize Jinja environment
        self.env = Environment(loader=FileSystemLoader(self.template_dir))
        # Register custom filters
        self.env.filters["basename"] = lambda path: os.path.basename(path) if path else ""

    def generate_markdown(self, dossier, output_path):
        """
        Renders the profile dossier as a clean, highly readable Markdown file.
        """
        metrics = dossier["metrics"]
        social = dossier["social_media"]
        
        lines = []
        lines.append(f"# Academic & Professional Dossier: {dossier['name']}")
        if dossier['affiliation']:
            lines.append(f"**Affiliation:** {dossier['affiliation']}")
        if dossier['orcid']:
            lines.append(f"**ORCID:** [{dossier['orcid']}](https://orcid.org/{dossier['orcid']})")
        lines.append("")
        
        if dossier['interests']:
            lines.append(f"**Research Fields:** {', '.join(dossier['interests'])}")
            lines.append("")

        # Metrics Table
        lines.append("## Bibliometric Metrics Summary")
        lines.append("| Metric | Value | Source (Best Match) |")
        lines.append("| :--- | :---: | :--- |")
        lines.append(f"| **H-Index** | {metrics['h_index']} | Combined |")
        lines.append(f"| **Total Citations** | {metrics['citations']:,} | Combined |")
        lines.append(f"| **Total Publications** | {metrics['publications']} | Combined |")
        if metrics['i10_index_google'] or metrics['i10_index_openalex']:
            i10 = max(metrics['i10_index_google'], metrics['i10_index_openalex'])
            lines.append(f"| **i10-Index** | {i10} | Scholar/OpenAlex |")
        lines.append("")

        # Citation history
        if metrics.get("citation_years"):
            lines.append("### Citation Trajectory (Last 10 Years)")
            lines.append("| Year | Citations | Works Published |")
            lines.append("| :--- | :---: | :---: |")
            for cy in metrics["citation_years"][-10:]:
                cbc = cy.get('cited_by_count') or 0
                lines.append(f"| {cy.get('year')} | {cbc:,} | {cy.get('works_count')} |")
            lines.append("")

        
        # Holistic Research Synthesis
        synth = dossier.get("research_synthesis")
        if synth and isinstance(synth, dict) and synth.get("evolution_of_work"):
            lines.append("## Intellectual Profile & Trajectory")
            lines.append("")
            lines.append("### Core Focus Areas")
            for fa in synth.get("core_focus_areas", []):
                lines.append(f"- {fa}")
            lines.append("")
            lines.append("### Evolution of Work")
            lines.append(synth.get("evolution_of_work", ""))
            lines.append("")
            lines.append("### Methodological Trends")
            for mt in synth.get("methodological_trends", []):
                lines.append(f"- {mt}")
            lines.append("")
            lines.append("### Overall Impact")
            lines.append(synth.get("overall_impact", ""))
            lines.append("")
            if synth.get("key_collaborators_or_institutions"):
                lines.append("### Notable Collaborations / Institutions")
                for kc in synth.get("key_collaborators_or_institutions", []):
                    lines.append(f"- {kc}")
                lines.append("")

        # Social Media Profile

        lines.append("## Professional & Social Media Dossier")
        
        # LinkedIn
        lines.append("### LinkedIn Presence")
        li_url = (social.get("linkedin") or {}).get("profile_url")
        if li_url:
            lines.append(f"**LinkedIn Profile:** [{li_url}]({li_url})")
        else:
            lines.append("*No public LinkedIn profile resolved.*")
        
        li_summary = (social.get("linkedin") or {}).get("summary")
        if li_summary:
            lines.append("")
            lines.append(li_summary)
        
        li_topics = (social.get("linkedin") or {}).get("key_topics")
        if li_topics:
            lines.append(f"\n*LinkedIn Themes:* {', '.join(li_topics)}")
        lines.append("")

        # Twitter
        lines.append("### Twitter/X Presence")
        tw_url = (social.get("twitter") or {}).get("profile_url")
        tw_handle = (social.get("twitter") or {}).get("handle")
        if tw_url:
            lines.append(f"**Twitter Profile:** [{tw_handle or tw_url}]({tw_url})")
        else:
            lines.append("*No public Twitter profile resolved.*")
            
        tw_summary = (social.get("twitter") or {}).get("summary")
        if tw_summary:
            lines.append("")
            lines.append(tw_summary)
            
        tw_topics = (social.get("twitter") or {}).get("key_topics")
        if tw_topics:
            lines.append(f"\n*Twitter Themes:* {', '.join(tw_topics)}")
        lines.append("")

        # Recent News
        news = social.get("recent_news_and_events")
        if news:
            lines.append("### Recent News & Public Media Mentions")
            lines.append(news)
            lines.append("")

        # Top co-authors
        if dossier.get("coauthors"):
            lines.append("## Top Collaborators & Co-authors")
            lines.append("| Collaborator | Co-authored Papers |")
            lines.append("| :--- | :---: |")
            for ca in dossier["coauthors"]:
                lines.append(f"| {ca['name']} | {ca['papers_count']} |")
            lines.append("")

        # Publications Detailed list
        lines.append("## Key Publications & Text Analysis")
        lines.append("*Publications are sorted by citation count. Analyzed papers contain a deep scientific review.*")
        lines.append("")

        for i, paper in enumerate(dossier["publications"]):
            lines.append(f"### {i+1}. {paper['title']}")
            lines.append(f"- **Year:** {paper.get('year') or 'N/A'} | **Venue:** {paper.get('venue') or 'Unknown'}")
            lines.append(f"- **Citations:** {paper.get('citations_max', 0)} (Max Combined)")
            
            links = []
            if paper.get("doi"):
                links.append(f"[DOI](https://doi.org/{paper['doi']})")
            if paper.get("arxiv"):
                links.append(f"[ArXiv](https://arxiv.org/abs/{paper['arxiv']})")
            if paper.get("oa_url"):
                links.append(f"[Open Access PDF]({paper['oa_url']})")
            if paper.get("pdf_path"):
                local_pdf_rel = f"pdfs/{os.path.basename(paper['pdf_path'])}"
                links.append(f"[Local PDF]({local_pdf_rel})")
            if links:
                lines.append(f"- **Links:** {' | '.join(links)}")
                
            if paper.get("authors"):
                lines.append(f"- **Authors:** {', '.join(paper['authors'][:10])}{' et al.' if len(paper['authors']) > 10 else ''}")
                
            tldr = paper.get("tldr")
            if tldr:
                lines.append(f"- **TL;DR:** *{tldr}*")

            # Deep Analysis
            analysis = paper.get("analysis")
            if analysis and "summary" in analysis:
                lines.append("")
                lines.append("> [!NOTE]")
                lines.append("> **Deep Text Analysis Summary:**")
                lines.append(f"> {analysis['summary']}")
                lines.append("> ")
                lines.append("> **Methodology:**")
                methodology = analysis.get("methodology") or analysis.get("method") or "N/A"
                lines.append(f"> {methodology}")
                lines.append("> ")
                
                datasets = analysis.get("datasets_used")
                if datasets:
                    lines.append("> **Datasets Used:**")
                    lines.append(f"> {', '.join(datasets)}")
                    lines.append("> ")
                
                lines.append("> **Core Contributions:**")
                conts = analysis.get("key_contributions") or analysis.get("contributions") or []
                for c in conts:
                    lines.append(f"> - {c}")
                lines.append("> ")
                
                outcome = analysis.get("outcome")
                if outcome and outcome != "N/A":
                    lines.append("> **Outcome:**")
                    lines.append(f"> {outcome}")
                    lines.append("> ")
                
                novelty = analysis.get("novelty")
                if novelty and novelty != "N/A":
                    lines.append("> **Novelty:**")
                    lines.append(f"> {novelty}")
                    lines.append("> ")
                
                patent = analysis.get("patent_details")
                if patent and patent != "N/A":
                    lines.append("> **Patent Information:**")
                    lines.append(f"> {patent}")
                    lines.append("> ")
                
                if analysis.get("limitations"):
                    lines.append("> **Limitations:**")
                    for l in analysis["limitations"]:
                        lines.append(f"> - {l}")
                    lines.append("> ")
                
                if analysis.get("future_work"):
                    lines.append("> **Future Work:**")
                    for fw in analysis["future_work"]:
                        lines.append(f"> - {fw}")
                    lines.append("> ")
            elif paper.get("abstract"):
                lines.append("")
                lines.append(f"**Abstract:** {paper['abstract'][:300]}...")
            
            lines.append("")
            lines.append("---")
            lines.append("")

        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            logger.info(f"Successfully generated Markdown report: {output_path}")
        except Exception as e:
            logger.error(f"Failed to generate Markdown report: {e}")

    def generate_html(self, dossier, output_path):
        """
        Renders the profile dossier as a gorgeous interactive HTML page.
        """
        try:
            template = self.env.get_template("report_template.html")
            
            social_safe = dossier.get("social_media") or {}
            if not social_safe.get("linkedin"):
                social_safe["linkedin"] = {}
            if not social_safe.get("twitter"):
                social_safe["twitter"] = {}
                
            html_content = template.render(
                dossier=dossier,
                metrics=dossier["metrics"],
                social=social_safe,
                synthesis=dossier.get("research_synthesis"),
                publications=dossier["publications"]
            )
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            logger.info(f"Successfully generated HTML report: {output_path}")
        except Exception as e:
            logger.error(f"Failed to generate HTML report: {e}")
            raise

    def generate_batch_html(self, batch_title, results, summary, output_path):
        """
        Renders the consolidated batch dossier as a gorgeous interactive dashboard.
        """
        try:
            template = self.env.get_template("batch_report_template.html")
            html_content = template.render(
                batch_title=batch_title,
                results=results,
                summary=summary
            )
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            logger.info(f"Successfully generated batch HTML report: {output_path}")
        except Exception as e:
            logger.error(f"Failed to generate batch HTML report: {e}")
            raise

    def generate_batch_markdown(self, batch_title, results, summary, output_path):
        """
        Generates a clean Markdown summary of the batch processing run.
        """
        try:
            lines = []
            lines.append(f"# {batch_title}")
            lines.append(f"Generated at: {summary['timestamp']}")
            lines.append("")
            lines.append("## Aggregated Statistics")
            lines.append(f"- **Total Researchers Processed:** {summary['total_targets']}")
            lines.append(f"- **Successful Collections:** {summary['successful_count']}")
            lines.append(f"- **Failed Collections:** {summary['failed_count']}")
            lines.append(f"- **Maximum H-Index:** {summary['max_h_index']}")
            lines.append(f"- **Total Citations:** {summary['total_citations']:,}")
            lines.append("")
            lines.append("## Batch Execution Results")
            lines.append("")
            lines.append("| Name | Affiliation | Status | H-Index | Citations | Publications | Report links |")
            lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :--- |")
            
            for r in results:
                name = r["name"]
                affil = r["affiliation"] or "N/A"
                status_str = "✅ Success" if r["status"] == "success" else "❌ Failed"
                if r["status"] == "success":
                    m = r["metrics"]
                    h_index = m["h_index"]
                    citations = f"{m['citations']:,}"
                    pubs = m["publications"]
                    links = f"[HTML](./{r['report_html']}) | [MD](./{r['report_md']})"
                else:
                    h_index = "N/A"
                    citations = "N/A"
                    pubs = "N/A"
                    links = f"N/A (Error: {r.get('error')})"
                
                lines.append(f"| **{name}** | {affil} | {status_str} | {h_index} | {citations} | {pubs} | {links} |")
                
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            logger.info(f"Successfully generated batch Markdown report: {output_path}")
        except Exception as e:
            logger.error(f"Failed to generate batch Markdown report: {e}")
            raise
