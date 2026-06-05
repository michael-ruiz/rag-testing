"""
retriever.py
------------
Loads the pre-built policy_index.npz and retrieves the top-K most
semantically similar crash policies for a given driving scene query.

Usage (standalone test):
    python retriever.py --query "A truck is braking hard in front of me"
    python retriever.py --query "Pedestrian crossing ahead" --top-k 3
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


class PolicyRetriever:
    def __init__(self, index_path: Path = Path("policy_index.npz")):
        if not index_path.exists():
            raise FileNotFoundError(
                f"Index not found: {index_path}\n"
                "Run `python build_index.py` first to build it."
            )
        data = np.load(index_path, allow_pickle=True)
        self.embeddings: np.ndarray = data["embeddings"]  # [N, D] float32, normalized
        self.policies: list[str] = data["policies"].tolist()
        self.vidnames: list[str] = data["vidnames"].tolist()
        model_name: str = data["model_name"][0]

        self._model = SentenceTransformer(model_name)
        print(f"[retriever] Loaded {len(self.policies)} policies | model: {model_name}")

    def retrieve(self, query: str, top_k: int = 1) -> list[dict]:
        """
        Retrieve the top-K most relevant crash policies for the given query.

        Returns a list of dicts with keys:
            rank (int), policy (str), vidname (str), score (float)
        """
        query_emb = self._model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )  # [1, D]

        # Cosine similarity = dot product (both sides are L2-normalized)
        scores: np.ndarray = (self.embeddings @ query_emb.T).squeeze()  # [N]

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            results.append(
                {
                    "rank": rank,
                    "policy": self.policies[idx],
                    "vidname": self.vidnames[idx],
                    "score": float(scores[idx]),
                }
            )
        return results


def main():
    parser = argparse.ArgumentParser(description="Test the policy retriever")
    parser.add_argument("--query", type=str, required=True, help="Driving scene description")
    parser.add_argument("--top-k", type=int, default=1, help="Number of results to return")
    parser.add_argument(
        "--index", type=Path, default=Path("policy_index.npz"), help="Path to index file"
    )
    args = parser.parse_args()

    retriever = PolicyRetriever(index_path=args.index)
    results = retriever.retrieve(args.query, top_k=args.top_k)

    print(f"\nQuery: {args.query}\n")
    for r in results:
        print(f"  Rank {r['rank']} (score={r['score']:.4f}) [{r['vidname']}]")
        print(f"  Policy: {r['policy']}\n")


if __name__ == "__main__":
    main()
