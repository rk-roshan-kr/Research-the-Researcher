import os
import tempfile
import unittest
from academic_pipeline.research_graph import (
    ResearchProgramGraph,
    ResearchProgramNode,
    ClaimNode,
    EvidenceNeedNode,
    ContradictionLink
)

class TestResearchProgramGraph(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_name = "test_research_graph.db"
        self.graph = ResearchProgramGraph(cache_dir=self.temp_dir.name, db_name=self.db_name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_save_and_load_program(self):
        node = ResearchProgramNode(
            id="prog_1",
            title="Continual Latent Dynamics",
            summary="A test program for testing.",
            university="EPFL",
            professor_name="Prof. X",
            last_refresh=123456.0
        )
        self.graph.save_program(node)
        
        # Reload
        self.graph.load_graph_from_db()
        self.assertTrue(self.graph.graph.has_node("prog_1"))
        data = self.graph.graph.nodes["prog_1"]
        self.assertEqual(data["title"], "Continual Latent Dynamics")
        self.assertEqual(data["professor_name"], "Prof. X")
        self.assertEqual(data["university"], "EPFL")

    def test_save_claim_and_linking(self):
        prog = ResearchProgramNode(
            id="prog_2",
            title="Continual Latent Dynamics",
            summary="A test program for testing.",
            university="EPFL",
            professor_name="Prof. X",
            last_refresh=123456.0
        )
        self.graph.save_program(prog)

        claim = ClaimNode(
            id="claim_1",
            text="CKA works well",
            type="theirs",
            confidence=0.9,
            paper_doi="10.1234/test",
            method="CKA",
            dataset="CIFAR10",
            metric="representational similarity",
            program_id="prog_2"
        )
        self.graph.save_claim(claim)

        self.graph.load_graph_from_db()
        self.assertTrue(self.graph.graph.has_node("claim_1"))
        self.assertTrue(self.graph.graph.has_edge("prog_2", "claim_1"))
        edge_data = self.graph.graph.edges["prog_2", "claim_1"]
        self.assertEqual(edge_data["relation"], "asserts")

    def test_save_contradiction(self):
        claim_yours = ClaimNode(
            id="c_yours",
            text="Curriculum ordering helps representation structure",
            type="ours",
            confidence=0.95,
            paper_doi="10.1111/yours",
            method="CKA",
            dataset="CEBRA",
            metric="similarity"
        )
        claim_theirs = ClaimNode(
            id="c_theirs",
            text="Curriculum ordering degrades representation structure",
            type="theirs",
            confidence=0.92,
            paper_doi="10.2222/theirs",
            method="RSA",
            dataset="MNIST",
            metric="similarity"
        )
        self.graph.save_claim(claim_yours)
        self.graph.save_claim(claim_theirs)

        link = ContradictionLink(
            id="link_1",
            my_claim_id="c_yours",
            their_claim_id="c_theirs",
            type="empirical",
            resolution_hypothesis="Difference due to metrics",
            severity=0.8,
            confidence=0.85,
            status="open"
        )
        self.graph.save_contradiction(link)

        self.graph.load_graph_from_db()
        self.assertTrue(self.graph.graph.has_edge("c_yours", "c_theirs"))
        edge_data = self.graph.graph.edges["c_yours", "c_theirs"]
        self.assertEqual(edge_data["relation"], "contradicts")
        self.assertEqual(edge_data["resolution_hypothesis"], "Difference due to metrics")

        contradictions = self.graph.get_contradictions()
        self.assertEqual(len(contradictions), 1)
        self.assertEqual(contradictions[0]["my_claim"], "Curriculum ordering helps representation structure")
        self.assertEqual(contradictions[0]["their_claim"], "Curriculum ordering degrades representation structure")

if __name__ == "__main__":
    unittest.main()
