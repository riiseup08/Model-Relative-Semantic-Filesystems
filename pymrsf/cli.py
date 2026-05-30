"""
pymrsf CLI — quick diagnostics and scoring from the command line.

Usage:
    pymrsf probe "The quick brown fox..."
    pymrsf score "chunk text" --query "what is this about?"
    pymrsf capabilities
"""
import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="pymrsf",
        description="pymrsf — Novelty-Aware RAG scoring CLI",
    )
    subparsers = parser.add_subparsers(dest="command")

    # probe subcommand
    p_probe = subparsers.add_parser("probe", help="Probe model knowledge of text")
    p_probe.add_argument("text", help="Text to probe")
    p_probe.add_argument("--verbose", "-v", action="store_true")
    p_probe.add_argument("--json", action="store_true", help="Output raw JSON")

    # score subcommand
    p_score = subparsers.add_parser("score", help="Score a single chunk")
    p_score.add_argument("chunk", help="Chunk text to score")
    p_score.add_argument("--query", "-q", default=None)
    p_score.add_argument("--verbose", "-v", action="store_true")
    p_score.add_argument("--relevance-cutoff", type=float, default=None)
    p_score.add_argument("--json", action="store_true", help="Output raw JSON")

    # capabilities subcommand
    subparsers.add_parser("capabilities", help="Show provider capabilities")

    args = parser.parse_args()

    if args.command == "probe":
        from pymrsf import probe, provider_capabilities
        if not provider_capabilities().get("supports_probe", False):
            print("[pymrsf] probe requires the local provider (PYMRSF_PROVIDER=local).", file=sys.stderr)
            sys.exit(1)
        result = probe(args.text, verbose=args.verbose)
        if args.json:
            print(json.dumps(result, indent=2))
        elif "error" in result:
            print(f"Error: {result['error']}", file=sys.stderr)
            sys.exit(1)
        else:
            print(f"knowledge_score : {result['knowledge_score']}/100")
            print(f"label           : {result['label']}")
            print(f"description     : {result['description']}")

    elif args.command == "score":
        from pymrsf import score_chunk
        result = score_chunk(
            args.chunk,
            query=args.query,
            verbose=args.verbose,
            relevance_cutoff=args.relevance_cutoff,
        )
        if args.json:
            print(json.dumps({k: v for k, v in result.items() if k != "chunk"}, indent=2))
        else:
            print(f"rag_score  : {result['rag_score']}/100")
            print(f"verdict    : {result['verdict']}")
            print(f"novelty    : {result['novelty_score']}/100")
            print(f"relevance  : {result['relevance_score']}/100")
            print(f"mode       : {result.get('scoring_mode', 'unknown')}")

    elif args.command == "capabilities":
        from pymrsf import provider_capabilities
        caps = provider_capabilities()
        print(f"{'Capability':<30} {'Available'}")
        print("-" * 42)
        for k, v in caps.items():
            print(f"  {k:<28} {v}")

    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
