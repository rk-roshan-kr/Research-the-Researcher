"""
crm_engine.py — Long-horizon relationship and interaction manager for the Research CRM.
"""

import logging
import uuid
import time
import csv
from datetime import datetime, timedelta
from typing import Optional, Literal

from academic_pipeline.investigation_schemas import (
    ResearchThread,
    ResearchInteraction,
    ResearchOutcome,
    ProfessorMemory,
    MemoryEvent,
)
from academic_pipeline.cache_manager import CacheManager
from academic_pipeline.investigation_engine import InvestigationEngine

logger = logging.getLogger(__name__)


class CRMEngine:
    """
    Manages the Research CRM layer including Threads, Interactions, Outcomes, and reminders.
    Directly interfaces with SQLite database through CacheManager.
    """

    # Dynamic follow-up scheduler rules (in days)
    FOLLOW_UP_INTERVALS = {
        "new_question": 14,
        "professor_reply": 7,
        "experiment_running": 21,
        "paper_update": 30,
    }

    def __init__(self, cache_manager: CacheManager, investigation_engine: InvestigationEngine):
        self._cache = cache_manager
        self._inv_engine = investigation_engine

    def _get_date_string(self, days_delta: int = 0) -> str:
        """Utility to get date string in YYYY-MM-DD format."""
        target_date = datetime.now() + timedelta(days=days_delta)
        return target_date.strftime("%Y-%m-%d")

    # ── Thread Management ─────────────────────────────────────────────────────

    def create_thread(self, prof_id: str, investigation_id: str, subject: str) -> ResearchThread:
        """Creates and saves a new ResearchThread."""
        thread = ResearchThread(
            id=str(uuid.uuid4()),
            professor_id=prof_id,
            investigation_id=investigation_id,
            subject=subject,
            status="active",
        )
        self._cache.save_thread(thread.to_dict())
        logger.info(f"Created research thread '{subject}' [{thread.id[:8]}]")
        return thread

    def get_thread(self, thread_id: str) -> Optional[ResearchThread]:
        data = self._cache.get_thread(thread_id)
        return ResearchThread.from_dict(data) if data else None

    def list_threads(self, prof_id: Optional[str] = None, investigation_id: Optional[str] = None) -> list[ResearchThread]:
        threads_data = self._cache.list_threads(prof_id, investigation_id)
        return [ResearchThread.from_dict(t) for t in threads_data]

    # ── Interaction Management ────────────────────────────────────────────────

    def create_interaction(
        self,
        thread_id: str,
        prof_id: str,
        investigation_id: str,
        blocker_id: str,
        question: str,
        interaction_number: int = 1,
    ) -> ResearchInteraction:
        """Drafts a new ResearchInteraction."""
        prof_mem = self._cache.get_professor_memory(prof_id)
        prof_name = prof_mem.get("professor_name", "Unknown Professor") if prof_mem else "Unknown Professor"

        interaction = ResearchInteraction(
            id=str(uuid.uuid4()),
            thread_id=thread_id,
            professor_name=prof_name,
            prof_id=prof_id,
            investigation_id=investigation_id,
            blocker_id=blocker_id,
            interaction_number=interaction_number,
            question=question,
            status="drafted",
            next_action="Review and send the drafted email/question.",
        )
        self._cache.save_interaction(interaction.to_dict())
        logger.info(f"Drafted interaction {interaction_number} for PI '{prof_name}' in thread '{thread_id[:8]}'")
        return interaction

    def get_interaction(self, interaction_id: str) -> Optional[ResearchInteraction]:
        data = self._cache.get_interaction(interaction_id)
        return ResearchInteraction.from_dict(data) if data else None

    def list_interactions(
        self,
        thread_id: Optional[str] = None,
        prof_id: Optional[str] = None,
        status_filter: Optional[str] = None
    ) -> list[ResearchInteraction]:
        data = self._cache.list_interactions(thread_id, prof_id, status_filter)
        return [ResearchInteraction.from_dict(i) for i in data]

    def mark_as_sent(self, interaction_id: str) -> ResearchInteraction:
        """Transitions an interaction from drafted -> sent and schedules follow-up."""
        interaction = self.get_interaction(interaction_id)
        if not interaction:
            raise ValueError(f"Interaction {interaction_id} not found.")

        interaction.status = "sent"
        interaction.updated_at = time.time()
        
        # Schedule follow-up dynamically
        # 1st email is a cold contact: 14 days. Subsequent ones: 7 days.
        days = self.FOLLOW_UP_INTERVALS["new_question"] if interaction.interaction_number == 1 else self.FOLLOW_UP_INTERVALS["professor_reply"]
        interaction.follow_up_date = self._get_date_string(days)
        interaction.next_action = f"Wait for reply. Follow up due on {interaction.follow_up_date}."

        self._cache.save_interaction(interaction.to_dict())

        # Update Professor Memory
        prof_mem_dict = self._cache.get_professor_memory(interaction.prof_id)
        if prof_mem_dict:
            prof_mem = ProfessorMemory.from_dict(prof_mem_dict)
            prof_mem.interaction_count += 1
            if prof_mem.status == "discovered":
                prof_mem.status = "contacted"
            
            # Log as standard event
            event = MemoryEvent(
                event_type="email_sent",
                notes=f"Sent interaction #{interaction.interaction_number}. Question: {interaction.question[:100]}...",
                investigation_id=interaction.investigation_id,
                blocker_id=interaction.blocker_id,
            )
            prof_mem.add_event(event)
            self._cache.save_professor_memory(interaction.prof_id, prof_mem.to_dict())

        # Keep thread updated
        thread = self.get_thread(interaction.thread_id)
        if thread and thread.status in ("dormant", "ignored"):
            thread.status = "active"
            thread.updated_at = time.time()
            self._cache.save_thread(thread.to_dict())

        logger.info(f"Marked interaction {interaction.id[:8]} as sent. Follow-up: {interaction.follow_up_date}")
        return interaction

    def log_reply(
        self,
        interaction_id: str,
        response_body: str,
        evidence_gained: Optional[str] = None,
        reply_quality: str = "short_reply",
    ) -> ResearchInteraction:
        """Logs a professor's email reply and resets the scheduling queue.

        Args:
            reply_quality: Classification of the reply depth.
                One of: no_reply | short_reply | technical_reply |
                advice_given | meeting_offered | collaboration_offer.
        """
        interaction = self.get_interaction(interaction_id)
        if not interaction:
            raise ValueError(f"Interaction {interaction_id} not found.")

        interaction.status = "replied"
        interaction.response = response_body
        interaction.evidence_gained = evidence_gained
        interaction.updated_at = time.time()
        
        # Next follow-up is scheduled for 5-7 days to draft a response/action
        days = self.FOLLOW_UP_INTERVALS["professor_reply"]
        interaction.follow_up_date = self._get_date_string(days)
        interaction.next_action = f"Process suggestion or reply. Log results or draft follow-up by {interaction.follow_up_date}."

        self._cache.save_interaction(interaction.to_dict())

        # Update Professor Memory
        prof_mem_dict = self._cache.get_professor_memory(interaction.prof_id)
        if prof_mem_dict:
            prof_mem = ProfessorMemory.from_dict(prof_mem_dict)
            prof_mem.reply_count += 1
            prof_mem.last_response_date = self._get_date_string()
            prof_mem.status = "in_conversation"
            
            notes = f"Received reply. Evidence gained: {evidence_gained}" if evidence_gained else "Received reply."
            event = MemoryEvent(
                event_type="new_paper_found" if "new paper" in (evidence_gained or "").lower() else "blocker_resolution_suggested",
                notes=notes,
                investigation_id=interaction.investigation_id,
                blocker_id=interaction.blocker_id,
                reply_quality=reply_quality,
            )
            prof_mem.add_event(event)
            self._cache.save_professor_memory(interaction.prof_id, prof_mem.to_dict())

        logger.info(f"Logged reply for interaction {interaction.id[:8]} from PI '{interaction.professor_name}' (quality={reply_quality})")
        return interaction

    # ── Outcome Tracking ──────────────────────────────────────────────────────

    def log_outcome(
        self,
        investigation_id: str,
        prof_id: str,
        blocker_id: str,
        suggestion: str,
        contribution_type: Literal[
            "paper_recommendation", "method_suggestion", "benchmark_suggestion",
            "dataset_suggestion", "theoretical_argument", "collaboration_offer", "other"
        ],
        implementation_notes: str,
        impact_score: float,
        resulting_artifact: str,
        resolve_associated_blocker: bool = False,
        strength: Literal["paper_mention", "pilot_result", "replicated_result", "published_result"] = "pilot_result",
    ) -> ResearchOutcome:
        """Records an experimental outcome from testing a PI's suggestion."""
        outcome = ResearchOutcome(
            id=str(uuid.uuid4()),
            investigation_id=investigation_id,
            professor_id=prof_id,
            suggestion=suggestion,
            contribution_type=contribution_type,
            implementation_notes=implementation_notes,
            impact_score=max(0.0, min(1.0, impact_score)),
            resulting_artifact=resulting_artifact,
        )
        self._cache.save_outcome(outcome.to_dict())

        # Update counts in Professor Memory
        prof_mem_dict = self._cache.get_professor_memory(prof_id)
        if prof_mem_dict:
            prof_mem = ProfessorMemory.from_dict(prof_mem_dict)
            prof_mem.outcome_count += 1
            
            event = MemoryEvent(
                event_type="evidence_recorded",
                notes=f"Tested suggestion ({contribution_type}): {suggestion[:50]}... Impact: {impact_score:.2f}",
                investigation_id=investigation_id,
                blocker_id=blocker_id,
                evidence_strength=strength,
                impact_score=impact_score,
            )
            prof_mem.add_event(event)
            self._cache.save_professor_memory(prof_id, prof_mem.to_dict())

        # Auto-resolve the blocker in investigation engine if specified
        if resolve_associated_blocker:
            try:
                prof_name = prof_mem_dict.get("professor_name", prof_id) if prof_mem_dict else prof_id
                self._inv_engine.resolve_blocker(
                    inv_id=investigation_id,
                    blocker_id=blocker_id,
                    resolved_by=prof_name,
                    evidence_strength=strength,
                    notes=f"Resolved via ResearchOutcome suggestion: {suggestion}. Results: {implementation_notes}",
                )
            except Exception as e:
                logger.warning(f"Failed to auto-resolve blocker {blocker_id}: {e}")

        # Update interactions in this thread to 'action_taken'
        interactions = self.list_interactions(prof_id=prof_id)
        matching_ints = [i for i in interactions if i.blocker_id == blocker_id and i.status == "replied"]
        for mint in matching_ints:
            mint.status = "action_taken"
            mint.next_action = "Action logged. Prepare next interaction with results."
            # Set dynamic follow-up to check running experiment progress or draft the update email (7 days)
            mint.follow_up_date = self._get_date_string(self.FOLLOW_UP_INTERVALS["professor_reply"])
            self._cache.save_interaction(mint.to_dict())

        logger.info(f"Logged outcome [{outcome.id[:8]}] for PI '{prof_id}' on blocker '{blocker_id[:8]}'")
        return outcome

    def list_outcomes(self, prof_id: Optional[str] = None, investigation_id: Optional[str] = None) -> list[ResearchOutcome]:
        data = self._cache.list_outcomes(prof_id, investigation_id)
        return [ResearchOutcome.from_dict(o) for o in data]

    # ── Reminders & Follow-Ups ────────────────────────────────────────────────

    def get_due_reminders(self) -> list[dict]:
        """Returns all sent interactions that require follow-up attention."""
        current_date = self._get_date_string()
        interactions = self.list_interactions(status_filter="sent")
        
        due = []
        for i in interactions:
            if i.follow_up_date and i.follow_up_date <= current_date:
                # Load investigation title
                inv = self._inv_engine._load(i.investigation_id)
                inv_title = inv.title if inv else "Unknown Investigation"
                
                due.append({
                    "interaction_id": i.id,
                    "thread_id": i.thread_id,
                    "professor_name": i.professor_name,
                    "prof_id": i.prof_id,
                    "investigation_id": i.investigation_id,
                    "investigation_title": inv_title,
                    "interaction_number": i.interaction_number,
                    "follow_up_date": i.follow_up_date,
                    "question": i.question,
                    "next_action": i.next_action,
                })
        return due

    # ── CSV Export ────────────────────────────────────────────────────────────

    def export_crm_to_csv(self, filepath: str) -> None:
        """Generates a CSV report summarizing the complete state of the CRM network."""
        threads = self.list_threads()
        
        with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "Professor ID",
                "Professor Name",
                "Institution",
                "Investigation Title",
                "Thread Subject",
                "Thread Status",
                "Latest Interaction Date",
                "Total Interactions",
                "Total Replies",
                "Relationship Strength",
                "Next Action",
                "Follow-up Due",
            ])
            
            for t in threads:
                # Load profile details
                prof_mem_dict = self._cache.get_professor_memory(t.professor_id)
                prof_name = prof_mem_dict.get("professor_name", "N/A") if prof_mem_dict else "N/A"
                institution = prof_mem_dict.get("institution", "N/A") if prof_mem_dict else "N/A"
                
                # Load investigation details
                inv = self._inv_engine._load(t.investigation_id)
                inv_title = inv.title if inv else "N/A"
                
                # Load interactions in this thread
                ints = self.list_interactions(thread_id=t.id)
                latest_int = ints[-1] if ints else None
                
                # Computations
                prof_mem = ProfessorMemory.from_dict(prof_mem_dict) if prof_mem_dict else None
                strength = prof_mem.get_relationship_strength() if prof_mem else 0.0
                
                writer.writerow([
                    t.professor_id,
                    prof_name,
                    institution,
                    inv_title,
                    t.subject,
                    t.status,
                    latest_int.updated_at if latest_int else "N/A",
                    prof_mem.interaction_count if prof_mem else 0,
                    prof_mem.reply_count if prof_mem else 0,
                    strength,
                    latest_int.next_action if latest_int else "Draft your first email.",
                    latest_int.follow_up_date if latest_int else "N/A",
                ])
                
        logger.info(f"Exported CRM database to {filepath}")
