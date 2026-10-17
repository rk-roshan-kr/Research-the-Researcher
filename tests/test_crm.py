import os
import unittest
import time
import tempfile
import shutil
from unittest.mock import MagicMock

from academic_pipeline.cache_manager import CacheManager
from academic_pipeline.investigation_schemas import (
    ProfessorMemory,
    MemoryEvent,
    ResearchThread,
    ResearchInteraction,
    ResearchOutcome,
    ResearchBlocker,
    InvestigationStream,
)
from academic_pipeline.investigation_engine import InvestigationEngine
from academic_pipeline.crm_engine import CRMEngine
from academic_pipeline.local_llm import LocalLLMClient


class TestResearchCRM(unittest.TestCase):
    """
    Unit test suite covering the Research CRM layer, databases, CRM engine states,
    and tension score gates.
    """

    def setUp(self):
        # Create a temporary directory for the SQLite database cache
        self.test_dir = tempfile.mkdtemp()
        self.cache = CacheManager(cache_dir=self.test_dir)
        
        # Setup mock LLM for InvestigationEngine
        self.inv_engine = InvestigationEngine(
            cache_manager=self.cache,
            ollama_model="mock-model",
            ollama_host="http://localhost:11434"
        )
        
        self.crm = CRMEngine(
            cache_manager=self.cache,
            investigation_engine=self.inv_engine
        )

    def tearDown(self):
        self.crm = None
        self.inv_engine = None
        self.cache = None
        import gc
        gc.collect()
        time.sleep(0.1)
        shutil.rmtree(self.test_dir)

    def test_professor_memory_relationship_strength(self):
        """Verify dynamic calculation of relationship strength in ProfessorMemory."""
        prof = ProfessorMemory(
            professor_name="Karl Friston",
            openalex_id="A1234",
            institution="UCL",
            status="discovered"
        )
        self.assertEqual(prof.get_relationship_strength(), 0.0)

        # Increment counts
        prof.interaction_count = 1
        # 1 interaction * 0.1 = 0.1
        self.assertEqual(prof.get_relationship_strength(), 0.1)

        prof.status = "in_conversation"
        prof.reply_count = 1
        # (1 reply * 0.4) + (1 interaction * 0.1) = 0.5
        self.assertEqual(prof.get_relationship_strength(), 0.5)

        prof.outcome_count = 1
        # (1 reply * 0.4) + (1 outcome * 0.3) + (1 interaction * 0.1) = 0.8
        self.assertEqual(prof.get_relationship_strength(), 0.8)

        # Test dormant scaling (should scale by 0.5)
        prof.status = "dormant"
        self.assertEqual(prof.get_relationship_strength(), 0.4)

        # Test ignored status (should yield 0.0)
        prof.status = "ignored"
        self.assertEqual(prof.get_relationship_strength(), 0.0)

    def test_professor_memory_collaboration_readiness_score(self):
        """Verify dynamic calculation of collaboration readiness score using split signals."""
        prof = ProfessorMemory(
            professor_name="Yann LeCun",
            openalex_id="A5678",
            institution="NYU",
            status="discovered",
            h_index=65,
            career_stage="full",
            last_publication_year=2025,
            student_supervision_signal=0.85
        )
        # Test basic field initialization
        self.assertEqual(prof.h_index, 65)
        self.assertEqual(prof.career_stage, "full")
        self.assertEqual(prof.last_publication_year, 2025)
        self.assertEqual(prof.student_supervision_signal, 0.85)

        # Test score calculation with mock lab_accessibility
        # status="discovered" (base = 0.05), reply_count=0, outcome_count=0
        # lab_accessibility=0.5 (accessibility_factor = 0.5 + 0.5 * 0.5 = 0.75)
        # score = 0.05 * 0.75 = 0.0375 -> round to 0.04
        self.assertAlmostEqual(prof.get_collaboration_readiness_score(lab_accessibility=0.5), 0.04, places=2)

        # Change status and replies
        prof.status = "contacted" # base + 0.10 -> 0.15
        prof.reply_count = 1       # +0.10 -> 0.25
        # score = 0.25 * 0.75 = 0.1875 -> round to 0.19
        self.assertAlmostEqual(prof.get_collaboration_readiness_score(lab_accessibility=0.5), 0.19, places=2)

        # In conversation and outcome count
        prof.status = "in_conversation" # base + 0.35 -> 0.40
        prof.reply_count = 2            # +0.20 -> 0.60
        prof.outcome_count = 1          # +0.15 -> 0.75
        # score = 0.75 * 0.75 = 0.5625 -> round to 0.56
        self.assertAlmostEqual(prof.get_collaboration_readiness_score(lab_accessibility=0.5), 0.56, places=2)

        # Test dormant scaling (halved)
        prof.status = "dormant"
        # base score computation:
        # base = 0.05 (no active status bonus since status is dormant)
        # score += reply_count * 0.10 (0.20) + outcome_count * 0.15 (0.15) = 0.40
        # accessibility factor = 0.75 -> 0.40 * 0.75 = 0.30
        # dormant status -> halved -> 0.15
        self.assertAlmostEqual(prof.get_collaboration_readiness_score(lab_accessibility=0.5), 0.15, places=2)

    def test_professor_memory_research_influence_score(self):
        """Verify dynamic calculation of research influence score in ProfessorMemory."""
        prof = ProfessorMemory(
            professor_name="Karl Friston",
            openalex_id="A1234",
            institution="UCL",
            status="discovered"
        )
        self.assertEqual(prof.get_research_influence_score(), 0.0)

        # Add non-influence events
        prof.add_event(MemoryEvent(event_type="dossier_viewed", notes="Viewed dossier"))
        prof.add_event(MemoryEvent(event_type="email_sent", notes="Sent first email"))
        self.assertEqual(prof.get_research_influence_score(), 0.0)

        # Add evidence_recorded events with impact scores
        prof.add_event(MemoryEvent(event_type="evidence_recorded", notes="Suggestion 1", impact_score=0.45))
        self.assertEqual(prof.get_research_influence_score(), 0.45)

        prof.add_event(MemoryEvent(event_type="evidence_recorded", notes="Suggestion 2", impact_score=0.33))
        # 0.45 + 0.33 = 0.78
        self.assertEqual(prof.get_research_influence_score(), 0.78)

        # Add an event with no impact score (or None)
        prof.add_event(MemoryEvent(event_type="evidence_recorded", notes="Suggestion 3", impact_score=None))
        self.assertEqual(prof.get_research_influence_score(), 0.78)

    def test_database_crud_threads_interactions_outcomes(self):
        """Test SQLite persistence for the new CRM schemas."""
        # Save and get ResearchThread
        t = ResearchThread(
            id="t-uuid-1",
            professor_id="prof-1",
            investigation_id="inv-1",
            subject="Curriculum learning discussion",
            status="active"
        )
        self.cache.save_thread(t.to_dict())
        t_loaded = self.cache.get_thread("t-uuid-1")
        self.assertIsNotNone(t_loaded)
        self.assertEqual(t_loaded["subject"], "Curriculum learning discussion")

        # Save and get ResearchInteraction
        i = ResearchInteraction(
            id="i-uuid-1",
            thread_id="t-uuid-1",
            professor_name="Martin Schrimpf",
            prof_id="prof-1",
            investigation_id="inv-1",
            blocker_id="blocker-1",
            interaction_number=1,
            question="Would your framework hold under C?",
            status="drafted"
        )
        self.cache.save_interaction(i.to_dict())
        i_loaded = self.cache.get_interaction("i-uuid-1")
        self.assertIsNotNone(i_loaded)
        self.assertEqual(i_loaded["question"], "Would your framework hold under C?")

        # Save and get ResearchOutcome
        o = ResearchOutcome(
            id="o-uuid-1",
            investigation_id="inv-1",
            professor_id="prof-1",
            suggestion="Try running CKA similarity metrics",
            contribution_type="method_suggestion",
            implementation_notes="Ran CKA over 3 curriculum conditions.",
            impact_score=0.85,
            resulting_artifact="benchmark code v1"
        )
        self.cache.save_outcome(o.to_dict())
        o_loaded = self.cache.get_outcome("o-uuid-1")
        self.assertIsNotNone(o_loaded)
        self.assertEqual(o_loaded["suggestion"], "Try running CKA similarity metrics")
        self.assertEqual(o_loaded["contribution_type"], "method_suggestion")

        # Test Listing
        threads = self.cache.list_threads(prof_id="prof-1")
        self.assertEqual(len(threads), 1)
        
        interactions = self.cache.list_interactions(thread_id="t-uuid-1")
        self.assertEqual(len(interactions), 1)

        outcomes = self.cache.list_outcomes(prof_id="prof-1")
        self.assertEqual(len(outcomes), 1)

    def test_crm_engine_relationship_lifecycle(self):
        """Test full CRM engine workflow from draft -> sent -> replied -> outcome logged."""
        # Create a professor memory first
        prof_mem = ProfessorMemory.new(
            professor_name="Alexander Mathis",
            institution="EPFL",
            openalex_id="prof-mathis-id"
        )
        self.cache.save_professor_memory(prof_mem.prof_id, prof_mem.to_dict())

        # Create an investigation with a blocker
        inv = self.inv_engine.create_investigation(
            title="TARS Exoplanets",
            description="ML models for light-curves",
            base_priority=0.8
        )
        # Bypassing validate_blocker_specificity for tests by using _save directly
        blocker = ResearchBlocker.new(
            description="Need a validated exoplanet transit light-curve dataset",
            blocker_type="dataset",
            severity=0.9,
            specificity_score=0.8
        )
        inv.blockers.append(blocker)
        inv.status = "blocked"
        self.inv_engine._save(inv)

        # 1. Create a Thread & Draft Interaction
        thread = self.crm.create_thread(prof_mem.prof_id, inv.id, "Light-curves dataset request")
        interaction = self.crm.create_interaction(
            thread_id=thread.id,
            prof_id=prof_mem.prof_id,
            investigation_id=inv.id,
            blocker_id=blocker.id,
            question="Do you have labeled exoplanet transits?",
            interaction_number=1
        )
        self.assertEqual(interaction.status, "drafted")

        # 2. Mark as Sent
        interaction = self.crm.mark_as_sent(interaction.id)
        self.assertEqual(interaction.status, "sent")
        self.assertIsNotNone(interaction.follow_up_date)
        
        # Verify professor memory incremented and transitioned
        prof_loaded = ProfessorMemory.from_dict(self.cache.get_professor_memory(prof_mem.prof_id))
        self.assertEqual(prof_loaded.interaction_count, 1)
        self.assertEqual(prof_loaded.status, "contacted")

        # 3. Log Reply
        interaction = self.crm.log_reply(
            interaction_id=interaction.id,
            response_body="Yes, we compiled a dataset. Take a look at doi:10.123/456.",
            evidence_gained="PI suggested transit dataset."
        )
        self.assertEqual(interaction.status, "replied")
        self.assertEqual(interaction.evidence_gained, "PI suggested transit dataset.")

        # Verify prof memory is updated to in_conversation
        prof_loaded = ProfessorMemory.from_dict(self.cache.get_professor_memory(prof_mem.prof_id))
        self.assertEqual(prof_loaded.reply_count, 1)
        self.assertEqual(prof_loaded.status, "in_conversation")
        self.assertEqual(prof_loaded.last_response_date, self.crm._get_date_string())

        # 4. Log Outcome (suggestion tested) and resolve blocker
        outcome = self.crm.log_outcome(
            investigation_id=inv.id,
            prof_id=prof_mem.prof_id,
            blocker_id=blocker.id,
            suggestion="Tested their transit dataset on our classifier",
            contribution_type="dataset_suggestion",
            implementation_notes="Achieved 95% accuracy using their dataset.",
            impact_score=0.9,
            resulting_artifact="light-curve benchmark v1",
            resolve_associated_blocker=True,
            strength="published_result"
        )
        self.assertEqual(outcome.suggestion, "Tested their transit dataset on our classifier")
        
        # Verify professor memory outcome count incremented
        prof_loaded = ProfessorMemory.from_dict(self.cache.get_professor_memory(prof_mem.prof_id))
        self.assertEqual(prof_loaded.outcome_count, 1)

        # Verify blocker gets resolved in investigation
        inv_loaded = self.inv_engine._load(inv.id)
        self.assertTrue(inv_loaded.blockers[0].resolved)
        self.assertEqual(inv_loaded.blockers[0].resolved_by, "Alexander Mathis")

        # Verify interaction is marked as action_taken
        interaction_loaded = self.crm.get_interaction(interaction.id)
        self.assertEqual(interaction_loaded.status, "action_taken")

    def test_tension_gate_threshold(self):
        """Verify the tension threshold gating behaves as expected."""
        client = LocalLLMClient(model="mock-model")
        client._chat_structured = MagicMock()

        # Mock LLM result below threshold
        client._chat_structured.return_value = {
            "has_tension": True,
            "tension_score": 0.45,
            "tension_description": "Weak topical alignment.",
            "intellectual_question": "Explain how curriculum learning works?",
            "rationale": "Topical connection only."
        }

        # Run with threshold 0.70 (default)
        result = client.detect_intellectual_tension(
            investigation_title="Developmental Cognition",
            blocker_description="Need evidence curriculum ordering changes representations",
            prof_paper_title="Curriculum learning in motor cortex",
            prof_paper_abstract="We show curriculum learning changes representation patterns.",
            tension_threshold=0.70
        )
        self.assertFalse(result["has_tension"])
        self.assertEqual(result["intellectual_question"], "")
        self.assertIn("below the required threshold", result["rationale"])

        # Mock LLM result above threshold
        client._chat_structured.return_value = {
            "has_tension": True,
            "tension_score": 0.88,
            "tension_description": "We use contradiction-driven curriculum, they use reward-driven.",
            "intellectual_question": "Would the representation changes persist if generated via contradiction-driven inquiry?",
            "rationale": "Bridges core structural tension."
        }

        result_pass = client.detect_intellectual_tension(
            investigation_title="Developmental Cognition",
            blocker_description="Need evidence curriculum ordering changes representations",
            prof_paper_title="Curriculum learning in motor cortex",
            prof_paper_abstract="We show curriculum learning changes representation patterns.",
            tension_threshold=0.70
        )
        self.assertTrue(result_pass["has_tension"])
        self.assertEqual(result_pass["tension_score"], 0.88)
        self.assertEqual(result_pass["intellectual_question"], "Would the representation changes persist if generated via contradiction-driven inquiry?")


if __name__ == "__main__":
    unittest.main()
