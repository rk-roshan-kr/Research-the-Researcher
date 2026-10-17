import argparse
import sys
from academic_pipeline.research_graph import ResearchProgramGraph

def main():
    parser = argparse.ArgumentParser(description="Research Program Market Map Explorer")
    parser.add_argument("--cache-dir", default="outputs", help="Directory where database files are cached")
    parser.add_argument("--method", help="Search programs by specific method (case-insensitive)")
    parser.add_argument("--dataset", help="Search programs by specific dataset (case-insensitive)")
    parser.add_argument("--metric", help="Search programs by specific metric (case-insensitive)")
    parser.add_argument("--list-contradictions", action="store_true", help="List all global contradictions detected across crawls")
    parser.add_argument("--list-programs", action="store_true", help="List all crawled programs and their PIs")

    args = parser.parse_args()
    
    try:
        graph = ResearchProgramGraph(cache_dir=args.cache_dir)
    except Exception as e:
        print(f"Error loading program graph database: {e}", file=sys.stderr)
        sys.exit(1)

    if args.list_contradictions:
        print("=== GLOBAL CONTRADICTIONS IN MARKET MAP ===")
        contradictions = graph.get_contradictions()
        if not contradictions:
            print("No global contradictions registered in database.")
        else:
            for i, c in enumerate(contradictions):
                print(f"{i+1}. Severity: {c['severity']:.2f} | Confidence: {c['confidence']:.2f} | Type: {c['type']}")
                print(f"   [Ours]  : \"{c['my_claim']}\" (DOI: {c['my_claim_doi']})")
                print(f"   [Their] : \"{c['their_claim']}\" (DOI: {c['their_claim_doi']})")
                print(f"   [Hypothesis]: {c['resolution_hypothesis']}")
                print("-" * 60)
        return

    if args.list_programs:
        print("=== REGISTERED RESEARCH PROGRAMS ===")
        programs = []
        for node, data in graph.graph.nodes(data=True):
            if data.get("type") == "program":
                programs.append(data)
        if not programs:
            print("No programs registered in database.")
        else:
            for p in programs:
                print(f"- {p['title']} (PI: {p['professor_name']} @ {p['university']})")
                print(f"  Summary: {p['summary']}")
        return

    # Check for search queries
    if args.method or args.dataset or args.metric:
        print(f"=== SEARCH RESULTS ===")
        found = False
        
        # Search claims matching constraints
        for node, data in graph.graph.nodes(data=True):
            if data.get("type") == "claim":
                m_match = not args.method or args.method.lower() in data.get("method", "").lower()
                d_match = not args.dataset or args.dataset.lower() in data.get("dataset", "").lower()
                mt_match = not args.metric or args.metric.lower() in data.get("metric", "").lower()
                
                if m_match and d_match and mt_match:
                    found = True
                    # Trace back to program
                    program_name = "Unknown Program"
                    pi_name = "Unknown PI"
                    univ = "Unknown University"
                    for u, v, edata in graph.graph.in_edges(node, data=True):
                        if edata.get("relation") == "asserts":
                            prog_node = graph.graph.nodes[u]
                            program_name = prog_node.get("title", "")
                            pi_name = prog_node.get("professor_name", "")
                            univ = prog_node.get("university", "")
                            break
                    
                    print(f"Claim: \"{data.get('text')}\"")
                    print(f"  Program: {program_name} (PI: {pi_name} @ {univ})")
                    print(f"  Method: {data.get('method')} | Dataset: {data.get('dataset')} | Metric: {data.get('metric')}")
                    print(f"  Paper DOI: {data.get('paper_doi')}")
                    print("-" * 50)
        if not found:
            print("No matching claims found.")
        return

    parser.print_help()

if __name__ == "__main__":
    main()
