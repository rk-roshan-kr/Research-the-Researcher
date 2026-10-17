import sqlite3
import os
import csv
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "leads.db"))

class LeadDatabase:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        # Ensure outputs folder exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_db()

    def init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS leads (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT UNIQUE,
                    institution TEXT,
                    position TEXT,
                    research_area TEXT,
                    faculty_url TEXT,
                    recent_paper TEXT,
                    why_selected TEXT,
                    campaign TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    -- Email Source Auditing
                    email_source_url TEXT,
                    email_source_type TEXT,
                    date_collected TEXT,
                    confidence INTEGER DEFAULT 0,
                    contact_permission TEXT DEFAULT 'unknown',
                    
                    -- CRM Metadata & Categorization
                    relationship TEXT DEFAULT 'cold',
                    cluster TEXT,
                    tier TEXT,
                    
                    -- Multi-Factor Scores
                    research_relevance INTEGER DEFAULT 0,
                    personal_purchase_probability INTEGER DEFAULT 0,
                    institutional_adoption_probability INTEGER DEFAULT 0,
                    influence_score INTEGER DEFAULT 0,
                    engagement_score INTEGER DEFAULT 0,
                    
                    -- Outreach State Machine
                    outreach_status TEXT DEFAULT 'Never Contacted',
                    selection_reason TEXT,
                    
                    -- Research Freshness
                    recent_publication_count INTEGER DEFAULT 0,
                    last_publication TEXT,
                    active_last_3_years INTEGER DEFAULT 0,
                    
                    -- Influence Telemetry Signals
                    google_scholar_citations INTEGER DEFAULT 0,
                    h_index INTEGER DEFAULT 0,
                    books_authored INTEGER DEFAULT 0,
                    course_instructor INTEGER DEFAULT 0,
                    newsletter INTEGER DEFAULT 0,
                    youtube INTEGER DEFAULT 0,
                    conference_speaker INTEGER DEFAULT 0,
                    editor INTEGER DEFAULT 0,
                    lab_director INTEGER DEFAULT 0
                )
            """)
            conn.commit()

    def add_lead(self, lead: Dict[str, Any]) -> bool:
        """Insert or replace a lead in the database with updated scoring models."""
        if not lead.get("email"):
            logger.warning(f"Skipping lead {lead.get('name')}: no email address provided.")
            return False
            
        # Calculate/derive CRM properties
        relevance = self.calculate_research_relevance(lead)
        influence = self.calculate_influence_score(lead)
        personal_prob, inst_prob = self.calculate_probabilities(lead, relevance, influence)
        tier = self.determine_tier(lead, influence)
        reason = self.generate_selection_reason(lead, tier, relevance)
        
        # Populate derived values back into lead dict
        lead["research_relevance"] = relevance
        lead["influence_score"] = influence
        lead["personal_purchase_probability"] = personal_prob
        lead["institutional_adoption_probability"] = inst_prob
        lead["tier"] = tier
        lead["selection_reason"] = reason
        lead["engagement_score"] = int((relevance * 0.4) + (influence * 0.3) + (max(personal_prob, inst_prob) * 0.3))
        
        if not lead.get("date_collected"):
            lead["date_collected"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        query = """
            INSERT INTO leads (
                name, email, institution, position, research_area, faculty_url,
                recent_paper, why_selected, campaign,
                email_source_url, email_source_type, date_collected, confidence, contact_permission,
                relationship, cluster, tier,
                research_relevance, personal_purchase_probability, institutional_adoption_probability,
                influence_score, engagement_score, outreach_status, selection_reason,
                recent_publication_count, last_publication, active_last_3_years,
                google_scholar_citations, h_index, books_authored, course_instructor,
                newsletter, youtube, conference_speaker, editor, lab_director
            ) VALUES (
                :name, :email, :institution, :position, :research_area, :faculty_url,
                :recent_paper, :why_selected, :campaign,
                :email_source_url, :email_source_type, :date_collected, :confidence, :contact_permission,
                :relationship, :cluster, :tier,
                :research_relevance, :personal_purchase_probability, :institutional_adoption_probability,
                :influence_score, :engagement_score, :outreach_status, :selection_reason,
                :recent_publication_count, :last_publication, :active_last_3_years,
                :google_scholar_citations, :h_index, :books_authored, :course_instructor,
                :newsletter, :youtube, :conference_speaker, :editor, :lab_director
            )
            ON CONFLICT(email) DO UPDATE SET
                name = coalesce(:name, name),
                institution = coalesce(:institution, institution),
                position = coalesce(:position, position),
                research_area = coalesce(:research_area, research_area),
                faculty_url = coalesce(:faculty_url, faculty_url),
                recent_paper = coalesce(:recent_paper, recent_paper),
                why_selected = coalesce(:why_selected, why_selected),
                campaign = coalesce(:campaign, campaign),
                email_source_url = coalesce(:email_source_url, email_source_url),
                email_source_type = coalesce(:email_source_type, email_source_type),
                confidence = coalesce(:confidence, confidence),
                contact_permission = coalesce(:contact_permission, contact_permission),
                relationship = coalesce(:relationship, relationship),
                cluster = coalesce(:cluster, cluster),
                tier = coalesce(:tier, tier),
                research_relevance = :research_relevance,
                personal_purchase_probability = :personal_purchase_probability,
                institutional_adoption_probability = :institutional_adoption_probability,
                influence_score = :influence_score,
                engagement_score = :engagement_score,
                outreach_status = coalesce(:outreach_status, outreach_status),
                selection_reason = :selection_reason,
                recent_publication_count = coalesce(:recent_publication_count, recent_publication_count),
                last_publication = coalesce(:last_publication, last_publication),
                active_last_3_years = coalesce(:active_last_3_years, active_last_3_years),
                google_scholar_citations = coalesce(:google_scholar_citations, google_scholar_citations),
                h_index = coalesce(:h_index, h_index),
                books_authored = coalesce(:books_authored, books_authored),
                course_instructor = coalesce(:course_instructor, course_instructor),
                newsletter = coalesce(:newsletter, newsletter),
                youtube = coalesce(:youtube, youtube),
                conference_speaker = coalesce(:conference_speaker, conference_speaker),
                editor = coalesce(:editor, editor),
                lab_director = coalesce(:lab_director, lab_director)
            WHERE :email IS NOT NULL AND :email != ''
        """



        db_dict = {
            "name": lead.get("name", ""),
            "email": lead.get("email"),
            "institution": lead.get("institution", ""),
            "position": lead.get("position", ""),
            "research_area": lead.get("research_area", ""),
            "faculty_url": lead.get("faculty_url", ""),
            "recent_paper": lead.get("recent_paper", ""),
            "why_selected": lead.get("why_selected", ""),
            "campaign": lead.get("campaign", "default"),
            "email_source_url": lead.get("email_source_url", ""),
            "email_source_type": lead.get("email_source_type", "unknown"),
            "date_collected": lead.get("date_collected"),
            "confidence": int(lead.get("confidence", 0) or 0),
            "contact_permission": lead.get("contact_permission", "unknown"),
            "relationship": lead.get("relationship", "cold"),
            "cluster": lead.get("cluster", lead.get("research_area", "General")),
            "tier": lead.get("tier", "PhD"),
            "research_relevance": lead.get("research_relevance", 0),
            "personal_purchase_probability": lead.get("personal_purchase_probability", 0),
            "institutional_adoption_probability": lead.get("institutional_adoption_probability", 0),
            "influence_score": lead.get("influence_score", 0),
            "engagement_score": lead.get("engagement_score", 0),
            "outreach_status": lead.get("outreach_status", "Never Contacted"),
            "selection_reason": lead.get("selection_reason", ""),
            "recent_publication_count": int(lead.get("recent_publication_count", 0) or 0),
            "last_publication": lead.get("last_publication", ""),
            "active_last_3_years": int(lead.get("active_last_3_years", 0) or 0),
            "google_scholar_citations": int(lead.get("google_scholar_citations", 0) or 0),
            "h_index": int(lead.get("h_index", 0) or 0),
            "books_authored": int(lead.get("books_authored", 0) or 0),
            "course_instructor": int(lead.get("course_instructor", 0) or 0),
            "newsletter": int(lead.get("newsletter", 0) or 0),
            "youtube": int(lead.get("youtube", 0) or 0),
            "conference_speaker": int(lead.get("conference_speaker", 0) or 0),
            "editor": int(lead.get("editor", 0) or 0),
            "lab_director": int(lead.get("lab_director", 0) or 0)
        }

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(query, db_dict)
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error adding lead {lead.get('name')}: {e}")
            return False

    # ────────────── Scoring Logics ──────────────

    def calculate_research_relevance(self, lead: Dict[str, Any]) -> int:
        """Score (0-100) research topic relevance using exact keyword checks."""
        score = 0
        keywords = ["continual learning", "active inference", "world models", 
                    "cognitive architecture", "reinforcement learning", 
                    "artificial intelligence", "deep learning", "agents",
                    "memory", "robotics", "neuroscience"]
        
        search_field = f"{lead.get('research_area', '')} {lead.get('recent_paper', '')} {lead.get('why_selected', '')}".lower()
        
        matches = [k for k in keywords if k in search_field]
        if matches:
            score += min(75, len(matches) * 20)
            
        # Active in last 3 years bonus
        if lead.get("active_last_3_years"):
            score += 25
            
        return min(100, score)

    def calculate_influence_score(self, lead: Dict[str, Any]) -> int:
        """Score (0-100) based on telemetry signals and default academic positions."""
        score = 0
        
        # 1. Base Score from Position
        pos = (lead.get("position") or "").lower()
        if "director" in pos or "head" in pos or "chair" in pos:
            score += 55
        elif "professor" in pos:
            score += 45
        elif "postdoc" in pos or "postdoctoral" in pos:
            score += 25
        elif "phd" in pos or "candidate" in pos or "student" in pos:
            score += 15
        elif "cto" in pos or "founder" in pos or "lead" in pos:
            score += 40
        else:
            score += 15

        # 2. Telemetry Modifiers
        citations = int(lead.get("google_scholar_citations", 0) or 0)
        score += min(20, citations // 200) # +1 point per 200 citations, max 20

        h_idx = int(lead.get("h_index", 0) or 0)
        score += min(15, h_idx) # +1 point per h_index, max 15

        if lead.get("books_authored"):
            score += min(10, int(lead.get("books_authored", 0)) * 5)

        if lead.get("lab_director"):
            score += 15
        if lead.get("editor"):
            score += 10
        if lead.get("course_instructor"):
            score += 10
        if lead.get("conference_speaker"):
            score += 8
        if lead.get("newsletter") or lead.get("youtube"):
            score += 10

        return min(100, score)

    def calculate_probabilities(self, lead: Dict[str, Any], relevance: int, influence: int) -> Tuple[int, int]:
        """Calculate (personal_purchase_probability, institutional_adoption_probability)."""
        # Base probabilities based on academic position and context
        pos = (lead.get("position") or "").lower()
        
        # 1. Personal Purchase (PhDs and Postdocs buy books, busy heads do not)
        pers_base = 50
        if "phd" in pos or "student" in pos:
            pers_base = 85
        elif "postdoc" in pos or "postdoctoral" in pos:
            pers_base = 75
        elif "professor" in pos:
            pers_base = 45
        elif "director" in pos or "head" in pos:
            pers_base = 25
        elif "engineer" in pos or "scientist" in pos:
            pers_base = 60
            
        personal_prob = pers_base + int(relevance * 0.15)
        
        # 2. Institutional Adoption (Professors and directors design curricula)
        inst_base = 10
        if "director" in pos or "head" in pos:
            inst_base = 90
        elif "professor" in pos:
            inst_base = 75
        elif "postdoc" in pos:
            inst_base = 20
        elif "phd" in pos or "student" in pos:
            inst_base = 5
            
        if lead.get("course_instructor"):
            inst_base += 15
        if lead.get("lab_director"):
            inst_base += 10
            
        inst_prob = inst_base + int(relevance * 0.1)

        return min(100, personal_prob), min(100, inst_prob)

    def determine_tier(self, lead: Dict[str, Any], influence: int) -> str:
        """Classify into named Tiers: Thought Leader, Faculty, Postdoc, PhD, Industry, Creator, Student."""
        pos = (lead.get("position") or "").lower()
        inst = (lead.get("institution") or "").lower()
        
        if influence >= 75 or "director" in pos or "head" in pos or "chair" in pos:
            return "Thought Leader"
            
        if "creator" in pos or "newsletter" in pos or "youtube" in pos or "blogger" in pos:
            return "Creator"
            
        if "professor" in pos or "faculty" in pos or "lecturer" in pos:
            return "Faculty"
            
        if "postdoc" in pos or "postdoctoral" in pos:
            return "Postdoc"
            
        if "phd" in pos or "candidate" in pos or "doctoral" in pos:
            return "PhD"
            
        # Check if industry institution
        industry_domains = ["google", "deepmind", "meta", "openai", "nvidia", "anthropic", "microsoft"]
        if any(ind in inst for ind in industry_domains) or "engineer" in pos or "scientist" in pos:
            return "Industry"
            
        if "student" in pos:
            return "Student"
            
        # Fallback based on influence
        if influence >= 40:
            return "Faculty"
        return "PhD"

    def generate_selection_reason(self, lead: Dict[str, Any], tier: str, relevance: int) -> str:
        """Generate selection reason explaining relevance, tier, and publications."""
        relevance_desc = f"High relevance ({relevance}%)" if relevance >= 70 else f"Moderate relevance ({relevance}%)"
        
        reasons = [f"Matched: {lead.get('research_area', 'AI')}", f"Tier: {tier}"]
        
        if lead.get("recent_paper"):
            reasons.append(f"Recent Paper: {lead.get('recent_paper')}")
            
        if lead.get("course_instructor"):
            reasons.append("Professor teaches related coursework")
            
        if lead.get("lab_director"):
            reasons.append("Directs a research laboratory")
            
        return " | ".join(reasons)

    # ────────────── DB Queries & Exporters ──────────────

    def get_leads(self, campaign: Optional[str] = None, min_relevance: int = 0) -> List[Dict[str, Any]]:
        query = "SELECT * FROM leads WHERE research_relevance >= ?"
        params = [min_relevance]
        if campaign:
            query += " AND campaign = ?"
            params.append(campaign)
        query += " ORDER BY research_relevance DESC"

        leads = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, params)
            for row in cursor.fetchall():
                leads.append(dict(row))
        return leads

    def export_to_csv(self, filepath: str, campaign: Optional[str] = None, min_relevance: int = 0) -> bool:
        leads = self.get_leads(campaign, min_relevance)
        if not leads:
            logger.warning("No leads found matching criteria to export.")
            return False

        headers = [
            "name", "email", "institution", "position", "research_area", 
            "faculty_url", "recent_paper", "why_selected", "campaign",
            "email_source_url", "email_source_type", "date_collected", "confidence", "contact_permission",
            "relationship", "cluster", "tier",
            "research_relevance", "personal_purchase_probability", "institutional_adoption_probability",
            "influence_score", "engagement_score", "outreach_status", "selection_reason",
            "recent_publication_count", "last_publication", "active_last_3_years",
            "google_scholar_citations", "h_index", "books_authored", "course_instructor",
            "newsletter", "youtube", "conference_speaker", "editor", "lab_director"
        ]

        try:
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            with open(filepath, mode="w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                for lead in leads:
                    filtered_lead = {h: lead.get(h, "") for h in headers}
                    writer.writerow(filtered_lead)
            return True
        except Exception as e:
            logger.error(f"Failed to export leads to {filepath}: {e}")
            return False

    def import_from_csv(self, filepath: str, campaign: str = "imported") -> int:
        if not os.path.exists(filepath):
            logger.error(f"CSV file does not exist: {filepath}")
            return 0

        count = 0
        try:
            with open(filepath, mode="r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    lead = {
                        "name": row.get("name") or row.get("Name") or "",
                        "email": row.get("email") or row.get("Email"),
                        "institution": row.get("institution") or row.get("Institution") or row.get("University") or "",
                        "position": row.get("position") or row.get("Position") or "",
                        "research_area": row.get("research_area") or row.get("Research Area") or "",
                        "faculty_url": row.get("faculty_url") or row.get("Faculty URL") or "",
                        "recent_paper": row.get("recent_paper") or row.get("Recent Paper") or "",
                        "why_selected": row.get("why_selected") or row.get("Why Selected") or "",
                        "campaign": campaign,
                        "email_source_url": row.get("email_source_url") or "",
                        "email_source_type": row.get("email_source_type") or "unknown",
                        "confidence": int(row.get("confidence") or 0),
                        "contact_permission": row.get("contact_permission") or "unknown",
                        "relationship": row.get("relationship") or "cold",
                        "cluster": row.get("cluster") or row.get("research_area") or "General",
                        "google_scholar_citations": int(row.get("google_scholar_citations") or 0),
                        "h_index": int(row.get("h_index") or 0),
                        "books_authored": int(row.get("books_authored") or 0),
                        "course_instructor": int(row.get("course_instructor") or 0),
                        "newsletter": int(row.get("newsletter") or 0),
                        "youtube": int(row.get("youtube") or 0),
                        "conference_speaker": int(row.get("conference_speaker") or 0),
                        "editor": int(row.get("editor") or 0),
                        "lab_director": int(row.get("lab_director") or 0),
                        "recent_publication_count": int(row.get("recent_publication_count") or 0),
                        "last_publication": row.get("last_publication") or "",
                        "active_last_3_years": int(row.get("active_last_3_years") or 0)
                    }
                    if not lead["name"]:
                        continue
                    if self.add_lead(lead):
                        count += 1
            return count
        except Exception as e:
            logger.error(f"Failed to import leads from {filepath}: {e}")
            return 0
