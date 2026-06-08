"""
eval_self_supervised.py
-----------------------
Self-supervised retrieval evaluation for the autonomous driving safety RAG pipeline.

Evaluation protocol:
  1. Load all valid triplets from crash_policies.jsonl.
  2. Randomly split 80% → index set / 20% → held-out test set (seed=42).
  3. Build a temporary in-memory index from the 80% index set only.
  4. For each held-out policy run THREE query conditions:
       a. latent_risk  → key_embeddings  (cross-field retrieval)
       b. mitigation   → key_embeddings  (cross-field retrieval)
       c. own trigger  → key_embeddings  (sanity check, should score ~1.0)
  5. Compute MRR@{1,5,10} and Recall@{1,5,10} for conditions a & b,
     plus average score of the correct match when it lands in top-10.
  6. Save full per-query details to eval_results_self_supervised.json.
  7. Print a clean summary table to stdout.

Does NOT modify or import retriever.py / build_index.py at the module level —
it reuses only the embedding logic (SentenceTransformer + L2 normalisation),
mirroring build_index.py exactly.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POLICIES_PATH = Path("crash_policies.jsonl")
OUTPUT_PATH = Path("eval_results_self_supervised.json")
MODEL_NAME = "all-MiniLM-L6-v2"
SPLIT_SEED = 42
TRAIN_RATIO = 0.80
TOP_K_VALUES = [1, 5, 10]

# ---------------------------------------------------------------------------
# Data loading  (mirrors build_index.load_policies exactly)
# ---------------------------------------------------------------------------


def load_policies(jsonl_path: Path) -> list[dict[str, str]]:
    """Return a list of validated policy dicts with keys: clip_id, trigger,
    latent_risk, mitigation.  Mirrors build_index.load_policies filtering."""
    required_keys = {"trigger", "latent_risk", "mitigation"}
    records: list[dict[str, str]] = []

    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            record = json.loads(raw)

            if record.get("error") is not None:
                continue
            if not required_keys.issubset(record.keys()):
                continue

            trigger = record["trigger"].strip()
            latent_risk = record["latent_risk"].strip()
            mitigation = record["mitigation"].strip()
            if not trigger or not latent_risk or not mitigation:
                continue

            clip_id = record.get("clip_id", record.get("vidname", "unknown"))
            records.append(
                {
                    "clip_id": clip_id,
                    "trigger": trigger,
                    "latent_risk": latent_risk,
                    "mitigation": mitigation,
                }
            )

    return records


# ---------------------------------------------------------------------------
# In-memory index
# ---------------------------------------------------------------------------


class InMemoryIndex:
    """Lightweight index that encodes triggers with L2-normalisation and
    supports dot-product (= cosine) similarity queries."""

    def __init__(self, policies: list[dict[str, str]], model: SentenceTransformer) -> None:
        self.policies = policies
        self.model = model

        triggers = [p["trigger"] for p in policies]
        print(f"[eval] Encoding {len(triggers)} index triggers …")
        self.key_embeddings: np.ndarray = model.encode(
            triggers,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,   # L2-normalise → cosine = dot product
            convert_to_numpy=True,
        )  # shape [N, D], float32

    def query(self, text: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Return top-k results sorted by descending cosine similarity."""
        q_emb = self.model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )  # [1, D]

        scores: np.ndarray = (self.key_embeddings @ q_emb.T).squeeze()  # [N]
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            results.append(
                {
                    "rank": rank,
                    "clip_id": self.policies[idx]["clip_id"],
                    "trigger": self.policies[idx]["trigger"],
                    "latent_risk": self.policies[idx]["latent_risk"],
                    "mitigation": self.policies[idx]["mitigation"],
                    "score": float(scores[idx]),
                }
            )
        return results


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def reciprocal_rank(results: list[dict], correct_clip_id: str, k: int) -> float:
    """Return 1/rank if correct_clip_id appears in top-k, else 0."""
    for r in results[:k]:
        if r["clip_id"] == correct_clip_id:
            return 1.0 / r["rank"]
    return 0.0


def recalled(results: list[dict], correct_clip_id: str, k: int) -> bool:
    return any(r["clip_id"] == correct_clip_id for r in results[:k])


def correct_score(results: list[dict], correct_clip_id: str) -> float | None:
    """Return the score of the correct match in results (any rank), else None."""
    for r in results:
        if r["clip_id"] == correct_clip_id:
            return r["score"]
    return None


# ---------------------------------------------------------------------------
# Evaluation for one query condition
# ---------------------------------------------------------------------------


def evaluate_condition(
    held_out: list[dict[str, str]],
    index: InMemoryIndex,
    query_field: str,
    condition_label: str,
) -> dict[str, Any]:
    """Run evaluation for a single query condition.

    Args:
        held_out:        List of held-out policy dicts.
        index:           InMemoryIndex built from the 80% set.
        query_field:     Which field to use as the query ('trigger',
                         'latent_risk', or 'mitigation').
        condition_label: Human-readable label for this condition.

    Returns a dict with aggregated metrics and per-query details.
    """
    max_k = max(TOP_K_VALUES)

    rr: dict[int, list[float]] = {k: [] for k in TOP_K_VALUES}
    hit: dict[int, list[int]] = {k: [] for k in TOP_K_VALUES}
    correct_scores: list[float] = []
    per_query: list[dict[str, Any]] = []

    for policy in held_out:
        query_text = policy[query_field]
        correct_id = policy["clip_id"]

        results = index.query(query_text, top_k=max_k)

        # Find rank of the correct clip_id (1-based), or None
        rank_of_correct: int | None = None
        score_of_correct: float | None = None
        for r in results:
            if r["clip_id"] == correct_id:
                rank_of_correct = r["rank"]
                score_of_correct = r["score"]
                break

        if score_of_correct is not None:
            correct_scores.append(score_of_correct)

        for k in TOP_K_VALUES:
            rr[k].append(reciprocal_rank(results, correct_id, k))
            hit[k].append(int(recalled(results, correct_id, k)))

        per_query.append(
            {
                "correct_clip_id": correct_id,
                "query_field": query_field,
                "query_text": query_text,
                "rank_of_correct": rank_of_correct,
                "score_of_correct": score_of_correct,
                "top_results": [
                    {"rank": r["rank"], "clip_id": r["clip_id"], "score": r["score"]}
                    for r in results[:max_k]
                ],
            }
        )

    n = len(held_out)
    mrr = {k: float(np.mean(rr[k])) for k in TOP_K_VALUES}
    recall = {k: float(np.mean(hit[k])) for k in TOP_K_VALUES}
    avg_correct_score = float(np.mean(correct_scores)) if correct_scores else None

    return {
        "condition": condition_label,
        "query_field": query_field,
        "n_evaluated": n,
        "mrr": mrr,
        "recall": recall,
        "avg_correct_score_top10": avg_correct_score,
        "n_correct_found_top10": len(correct_scores),
        "per_query": per_query,
    }


# ---------------------------------------------------------------------------
# Pretty-print summary table
# ---------------------------------------------------------------------------


def print_summary(results_by_condition: list[dict[str, Any]]) -> None:
    k_cols = TOP_K_VALUES  # [1, 5, 10]

    # Column widths
    cw = 40   # condition label
    mw = 10   # metric cell

    def sep() -> None:
        print("+" + "-" * (cw + 2) + "+" + ("+".join(["-" * (mw + 2)] * (len(k_cols) * 2 + 2))) + "+")

    def row(*cells: str) -> None:
        widths = [cw] + [mw] * (len(k_cols) * 2 + 2)
        parts = [f" {str(c).center(w)} " for c, w in zip(cells, widths)]
        print("|" + "|".join(parts) + "|")

    header_cells = (
        ["Condition"]
        + [f"MRR@{k}" for k in k_cols]
        + [f"Recall@{k}" for k in k_cols]
        + ["AvgScore", "N"]
    )

    print()
    print("=" * 90)
    print("  SELF-SUPERVISED RETRIEVAL EVALUATION")
    print("=" * 90)
    sep()
    row(*header_cells)
    sep()

    for cond in results_by_condition:
        label = cond["condition"]
        mrr_cells = [f"{cond['mrr'][k]:.4f}" for k in k_cols]
        recall_cells = [f"{cond['recall'][k]:.4f}" for k in k_cols]
        avg_score = (
            f"{cond['avg_correct_score_top10']:.4f}"
            if cond["avg_correct_score_top10"] is not None
            else "  N/A  "
        )
        n_cell = str(cond["n_evaluated"])
        row(label, *mrr_cells, *recall_cells, avg_score, n_cell)

    sep()
    print()

    for cond in results_by_condition:
        n_found = cond["n_correct_found_top10"]
        n_total = cond["n_evaluated"]
        pct = 100.0 * n_found / n_total if n_total else 0.0
        print(
            f"  [{cond['condition']}]  "
            f"correct in top-10: {n_found}/{n_total} ({pct:.1f}%)"
        )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # 1. Load policies
    print(f"[eval] Loading policies from {POLICIES_PATH} …")
    all_policies = load_policies(POLICIES_PATH)
    print(f"[eval] Loaded {len(all_policies)} valid policies.")

    if len(all_policies) < 5:
        raise RuntimeError("Too few policies to evaluate — check crash_policies.jsonl.")

    # 2. Split 80/20
    rng = random.Random(SPLIT_SEED)
    shuffled = all_policies[:]
    rng.shuffle(shuffled)

    split = math.ceil(len(shuffled) * TRAIN_RATIO)
    index_set = shuffled[:split]
    held_out = shuffled[split:]

    print(
        f"[eval] Split → index set: {len(index_set)}  |  held-out: {len(held_out)}"
        f"  (seed={SPLIT_SEED}, ratio={TRAIN_RATIO:.0%})"
    )

    # 3. Build in-memory index from index_set only
    print(f"[eval] Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    index = InMemoryIndex(index_set, model)

    # 4 & 5. Evaluate three conditions
    print("[eval] Evaluating condition (a): latent_risk → triggers …")
    cond_a = evaluate_condition(
        held_out, index,
        query_field="latent_risk",
        condition_label="latent_risk → trigger",
    )

    print("[eval] Evaluating condition (b): mitigation → triggers …")
    cond_b = evaluate_condition(
        held_out, index,
        query_field="mitigation",
        condition_label="mitigation → trigger",
    )

    # 6. Sanity check: trigger → triggers
    print("[eval] Evaluating condition (c): trigger → trigger (sanity check) …")
    cond_c = evaluate_condition(
        held_out, index,
        query_field="trigger",
        condition_label="Sanity check (trigger→trigger)",
    )

    all_conditions = [cond_a, cond_b, cond_c]

    # 7. Print summary
    print_summary(all_conditions)

    # 8. Save full results
    output = {
        "meta": {
            "policies_path": str(POLICIES_PATH),
            "model": MODEL_NAME,
            "split_seed": SPLIT_SEED,
            "train_ratio": TRAIN_RATIO,
            "n_total": len(all_policies),
            "n_index": len(index_set),
            "n_held_out": len(held_out),
            "top_k_values": TOP_K_VALUES,
        },
        "conditions": all_conditions,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)

    print(f"[eval] Full results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
