"""
build_index.py
--------------
Embeds all policy strings from crash_policies.jsonl using
sentence-transformers and saves a fast-lookup index to policy_index.npz.

Usage:
    python build_index.py
    python build_index.py --policies crash_policies.jsonl --out policy_index.npz
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def load_policies(jsonl_path: Path) -> tuple[list[str], list[str]]:
    """Load policy strings and video IDs from a JSONL file."""
    policies, vidnames = [], []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("error") is not None:
                continue  # skip errored entries
            policy = record.get("policy", "").strip()
            if policy:
                policies.append(policy)
                vidnames.append(record.get("vidname", "unknown"))
    return policies, vidnames


def build_index(
    jsonl_path: Path,
    out_path: Path,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
) -> None:
    print(f"[build_index] Loading policies from: {jsonl_path}")
    policies, vidnames = load_policies(jsonl_path)
    print(f"[build_index] Found {len(policies)} valid policies.")

    print(f"[build_index] Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    print("[build_index] Encoding policies (this may take a minute on first run)...")
    embeddings = model.encode(
        policies,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # cosine sim = dot product after normalization
        convert_to_numpy=True,
    )

    np.savez(
        out_path,
        embeddings=embeddings.astype(np.float32),
        policies=np.array(policies, dtype=object),
        vidnames=np.array(vidnames, dtype=object),
        model_name=np.array([model_name]),
    )
    print(f"[build_index] Index saved to: {out_path}  ({len(policies)} entries)")


def main():
    parser = argparse.ArgumentParser(description="Build policy vector index")
    parser.add_argument(
        "--policies",
        type=Path,
        default=Path("crash_policies.jsonl"),
        help="Path to crash_policies.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("policy_index.npz"),
        help="Output path for the index file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence-transformer model name",
    )
    args = parser.parse_args()

    if not args.policies.exists():
        raise FileNotFoundError(f"Policies file not found: {args.policies}")

    build_index(args.policies, args.out, model_name=args.model)


if __name__ == "__main__":
    main()
