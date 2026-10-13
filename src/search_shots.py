import argparse
from pathlib import Path

from semantic_search import ShotSemanticSearch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search analyzed film shots using natural-language text."
    )
    parser.add_argument("query", help="Natural-language search query")
    parser.add_argument(
        "--analysis",
        default="data/output/analysis.json",
        help="Path to analysis.json",
    )
    parser.add_argument(
        "--index",
        default="data/output/semantic_index.npz",
        help="Path to semantic index",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to show",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild image embeddings before searching",
    )

    args = parser.parse_args()

    analysis_path = Path(args.analysis)
    index_path = Path(args.index)

    search = ShotSemanticSearch()

    if args.rebuild_index or not index_path.exists():
        print("Building semantic index...")
        search.build_index(
            analysis_path=analysis_path,
            index_path=index_path,
        )

    results = search.search(
        query=args.query,
        analysis_path=analysis_path,
        index_path=index_path,
        top_k=args.top_k,
    )

    print(f'\nResults for "{args.query}":\n')

    for result in results:
        print(
            f'{result["rank"]}. Shot {result["shot_id"]:03d} '
            f'| score={result["score"]:.3f} '
            f'| {result["start"]:.2f}s-{result["end"]:.2f}s '
            f'| {result["thumbnail"]}'
        )


if __name__ == "__main__":
    main()
