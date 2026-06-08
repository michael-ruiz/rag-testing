"""
eval_causal_knockout.py
-----------------------
Causal knockout evaluation for the autonomous driving safety RAG pipeline.

Tests whether retrieved crash knowledge causally changes the pipeline output,
or whether the LLM produces the same response regardless of memory.

Protocol:
  For each scene caption, run the inference pipeline TWICE:

  Condition A — WITH memory:
    Retrieve top-1 policy from policy_index.npz, inject trigger/latent_risk/
    mitigation into infer.py, record gated output.

  Condition B — WITHOUT memory (knockout):
    Pass empty policy strings with a null-policy system prompt override.
    No retrieval is performed.

  Compare outputs to measure:
    - Action divergence rate (string inequality + semantic similarity)
    - Gate influence (divergence rate split by OPEN vs CLOSED gate)
    - Retrieval score distribution by gate status
    - Gate prediction accuracy vs expected_gate labels (if present)

Usage:
    python eval_causal_knockout.py
    python eval_causal_knockout.py --scenes eval_scenes.jsonl --top-k 1 --delay 1.5
    python eval_causal_knockout.py --max-scenes 10 --no-covla-fallback
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Imports from existing pipeline modules (read-only — do NOT modify those files)
# ---------------------------------------------------------------------------
from retriever import PolicyRetriever
from infer import run_inference, SYSTEM_PROMPT, extract_json, validate_output, DEFAULT_MODEL

load_dotenv()

# ---------------------------------------------------------------------------
# Constants & defaults
# ---------------------------------------------------------------------------

SCENES_PATH = Path("eval_scenes.jsonl")
INDEX_PATH = Path("policy_index.npz")
OUTPUT_PATH = Path("eval_results_knockout.json")
EMBED_MODEL = "all-MiniLM-L6-v2"




# ---------------------------------------------------------------------------
# Scene loading
# ---------------------------------------------------------------------------


def load_eval_scenes(
    scenes_path: Path,
    max_samples: int | None,
    use_covla_fallback: bool,
    hf_token: str | None,
) -> list[dict[str, Any]]:
    """Load scenes with priority: eval_scenes.jsonl → CoVLA fallback."""

    if scenes_path.exists():
        print(f"[knockout] Loading scenes from {scenes_path} …")
        scenes: list[dict[str, Any]] = []
        with open(scenes_path, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if max_samples is not None and len(scenes) >= max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                caption = record.get("caption", "").strip()
                if not caption:
                    continue
                scenes.append(
                    {
                        "scene_id": record.get("scene_id", f"scene_{i:04d}"),
                        "caption": caption,
                        "expected_risk": record.get("expected_risk", None),
                        "expected_gate": record.get("expected_gate", None),
                    }
                )
        print(f"[knockout] Loaded {len(scenes)} scenes from {scenes_path}.")
        return scenes

    # Fallback: CoVLA
    if not use_covla_fallback:
        raise FileNotFoundError(
            f"{scenes_path} not found and CoVLA fallback is disabled. "
            "Create eval_scenes.jsonl or run without --no-covla-fallback."
        )

    print(f"[knockout] {scenes_path} not found — falling back to CoVLA (first 50 captions) …")
    try:
        from load_covla import load_captions
    except ImportError:
        raise ImportError("load_covla.py not found in project directory.")

    limit = min(max_samples or 50, 50)
    scenes = []
    for i, rec in enumerate(load_captions(hf_token=hf_token, max_samples=limit)):
        if max_samples is not None and len(scenes) >= max_samples:
            break
        caption = rec.get("rich_caption", "").strip()
        if not caption:
            continue
        scenes.append(
            {
                "scene_id": rec.get("frame_id", f"covla_{i:04d}"),
                "caption": caption,
                "expected_risk": None,
                "expected_gate": None,
            }
        )
    print(f"[knockout] Loaded {len(scenes)} scenes from CoVLA.")
    return scenes





# ---------------------------------------------------------------------------
# Semantic similarity helper
# ---------------------------------------------------------------------------


def embed_texts(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    return model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalised vectors."""
    return float(np.dot(a, b))


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _std(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) >= 2 else None


# ---------------------------------------------------------------------------
# Results aggregation
# ---------------------------------------------------------------------------


def compute_metrics(scene_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute all aggregate metrics from per-scene results."""
    n = len(scene_results)

    # Gate counts
    open_scenes = [r for r in scene_results if r["cond_a"]["gate_status"] == "OPEN"]
    closed_scenes = [r for r in scene_results if r["cond_a"]["gate_status"] == "CLOSED"]

    def divergence_stats(subset: list[dict]) -> dict:
        if not subset:
            return {"n": 0, "string_divergence_rate": None, "semantic_divergence_rate": None}
        string_div = [r["divergence_string"] for r in subset]
        sem_sim = [r["semantic_similarity"] for r in subset]
        # Semantic divergence: similarity < 0.90 (strong paraphrase threshold)
        sem_div_90 = [s < 0.90 for s in sem_sim]
        sem_div_85 = [s < 0.85 for s in sem_sim]
        return {
            "n": len(subset),
            "string_divergence_rate": sum(string_div) / len(string_div),
            "semantic_divergence_rate": sum(sem_div_90) / len(sem_div_90),
            "semantic_divergence_rate_85": sum(sem_div_85) / len(sem_div_85),
            "mean_semantic_similarity": _mean(sem_sim),
        }

    # Retrieval score distribution
    open_scores = [r["retrieval_score"] for r in open_scenes if r["retrieval_score"] is not None]
    closed_scores = [r["retrieval_score"] for r in closed_scenes if r["retrieval_score"] is not None]

    # Gate accuracy (only for scenes with expected_gate label)
    labeled = [r for r in scene_results if r["expected_gate"] is not None]
    gate_accuracy: float | None = None
    if labeled:
        correct = sum(
            1 for r in labeled
            if r["cond_a"]["gate_status"] == r["expected_gate"]
        )
        gate_accuracy = correct / len(labeled)

    # Overall semantic similarity
    all_sims = [r["semantic_similarity"] for r in scene_results]
    pre_gated_count = sum(1 for r in scene_results if r.get("pre_gated"))

    return {
        "n_total": n,
        "n_open": len(open_scenes),
        "n_closed": len(closed_scenes),
        "open_rate": len(open_scenes) / n if n else None,
        "closed_rate": len(closed_scenes) / n if n else None,
        "divergence_open": divergence_stats(open_scenes),
        "divergence_closed": divergence_stats(closed_scenes),
        "divergence_all": divergence_stats(scene_results),
        "retrieval_scores": {
            "open": {
                "mean": _mean(open_scores),
                "std": _std(open_scores),
                "n": len(open_scores),
            },
            "closed": {
                "mean": _mean(closed_scores),
                "std": _std(closed_scores),
                "n": len(closed_scores),
            },
        },
        "gate_accuracy_vs_expected": gate_accuracy,
        "n_labeled": len(labeled),
        "overall_mean_semantic_similarity": _mean(all_sims),
        "pre_gated_count": pre_gated_count,
    }


# ---------------------------------------------------------------------------
# Summary table printer
# ---------------------------------------------------------------------------


def print_summary(metrics: dict[str, Any], scene_results: list[dict[str, Any]]) -> None:
    n = metrics["n_total"]
    n_open = metrics["n_open"]
    n_closed = metrics["n_closed"]

    def fmt_rate(v: float | None, denom: int = 0) -> str:
        if v is None:
            return "N/A"
        return f"{v:.1%}" + (f"  ({int(round(v * denom))}/{denom})" if denom else "")

    def fmt_f(v: float | None, decimals: int = 4) -> str:
        return f"{v:.{decimals}f}" if v is not None else "N/A"

    div_open = metrics["divergence_open"]
    div_closed = metrics["divergence_closed"]
    div_all = metrics["divergence_all"]
    rs = metrics["retrieval_scores"]
    gate_acc = metrics["gate_accuracy_vs_expected"]

    print()
    print("=" * 70)
    print("  CAUSAL KNOCKOUT EVALUATION — SUMMARY")
    print("=" * 70)
    print(f"  Total scenes evaluated          : {n}")
    pre_gated = metrics.get('pre_gated_count', 0)
    print(f"  Pre-gated by threshold (<{metrics.get('sim_threshold', 0.75):.2f}) : {pre_gated}  ({fmt_rate(pre_gated/n if n else 0)})")
    print(f"  Passed through to LLM           : {n - pre_gated}  ({fmt_rate((n - pre_gated)/n if n else 0)})")
    print(f"  Gate OPEN  (A=OPEN)             : {n_open}  ({fmt_rate(metrics['open_rate'])})")
    print(f"  Gate CLOSED (A=CLOSED)          : {n_closed}  ({fmt_rate(metrics['closed_rate'])})")
    print()
    print("  ── Action Divergence (A vs B) ──────────────────────────────────")
    print(f"  String divergence  — OPEN  gates: {fmt_rate(div_open['string_divergence_rate'],  div_open['n'])}")
    print(f"  String divergence  — CLOSED gates: {fmt_rate(div_closed['string_divergence_rate'], div_closed['n'])}")
    print(f"  String divergence  — ALL   scenes: {fmt_rate(div_all['string_divergence_rate'],  div_all['n'])}")
    print()
    print(f"  Semantic divergence (<0.90)— OPEN  gates: {fmt_rate(div_open['semantic_divergence_rate'],  div_open['n'])}")
    print(f"  Semantic divergence (<0.90)— CLOSED gates: {fmt_rate(div_closed['semantic_divergence_rate'], div_closed['n'])}")
    print(f"  Semantic divergence (<0.90)— ALL   scenes: {fmt_rate(div_all['semantic_divergence_rate'],  div_all['n'])}")
    print()
    print(f"  Semantic divergence (<0.85)— OPEN  gates: {fmt_rate(div_open['semantic_divergence_rate_85'],  div_open['n'])}")
    print(f"  Semantic divergence (<0.85)— CLOSED gates: {fmt_rate(div_closed['semantic_divergence_rate_85'], div_closed['n'])}")
    print(f"  Semantic divergence (<0.85)— ALL   scenes: {fmt_rate(div_all['semantic_divergence_rate_85'],  div_all['n'])}")
    print()
    print(f"  Overall mean semantic similarity : {fmt_f(metrics['overall_mean_semantic_similarity'])}")
    print()
    print("  ── Retrieval Score Distribution ────────────────────────────────")
    print(f"  Mean score — OPEN  gates        : {fmt_f(rs['open']['mean'])}  (std={fmt_f(rs['open']['std'])})")
    print(f"  Mean score — CLOSED gates       : {fmt_f(rs['closed']['mean'])}  (std={fmt_f(rs['closed']['std'])})")
    if gate_acc is not None:
        print()
        print(f"  Gate prediction accuracy        : {fmt_rate(gate_acc, metrics['n_labeled'])}  ({metrics['n_labeled']} labeled)")
    print("=" * 70)
    print()


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Causal knockout evaluation for the RAG pipeline")
    parser.add_argument(
        "--scenes", type=Path, default=SCENES_PATH,
        help="Path to eval_scenes.jsonl (default: eval_scenes.jsonl)"
    )
    parser.add_argument(
        "--index", type=Path, default=INDEX_PATH,
        help="Path to policy_index.npz (default: policy_index.npz)"
    )
    parser.add_argument(
        "--top-k", type=int, default=1,
        help="Number of policies to retrieve in condition A (default: 1)"
    )
    parser.add_argument(
        "--delay", type=float, default=1.0,
        help="Seconds to wait between scene evaluations (default: 1.0)"
    )
    parser.add_argument(
        "--sim-threshold", type=float, default=0.75,
        help="Retrieval score threshold for pre-gating condition A (default: 0.75)"
    )
    parser.add_argument(
        "--max-scenes", type=int, default=None,
        help="Maximum number of scenes to evaluate (default: all)"
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Groq model ID (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--token", type=str, default=None,
        help="Groq API token (overrides .env GROQ_API_KEY)"
    )
    parser.add_argument(
        "--no-covla-fallback", action="store_true",
        help="Disable CoVLA fallback if eval_scenes.jsonl is not found"
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_PATH,
        help=f"Output JSON path (default: {OUTPUT_PATH})"
    )
    args = parser.parse_args()

    api_key = args.token or os.getenv("GROQ_API_KEY")
    hf_token = os.getenv("HF_TOKEN")

    # ── 1. Load scenes ──────────────────────────────────────────────────────
    scenes = load_eval_scenes(
        scenes_path=args.scenes,
        max_samples=args.max_scenes,
        use_covla_fallback=not args.no_covla_fallback,
        hf_token=hf_token,
    )

    if not scenes:
        raise RuntimeError("No scenes loaded — cannot run evaluation.")

    # ── 2. Load retriever (condition A) ────────────────────────────────────
    print(f"[knockout] Loading policy index from {args.index} …")
    retriever = PolicyRetriever(index_path=args.index)

    # ── 3. Load embedding model for semantic similarity ─────────────────────
    print(f"[knockout] Loading embedding model ({EMBED_MODEL}) for semantic similarity …")
    embed_model = SentenceTransformer(EMBED_MODEL)

    # ── 4. Run evaluation loop ──────────────────────────────────────────────
    scene_results: list[dict[str, Any]] = []
    n = len(scenes)

    print(f"\n[knockout] Starting evaluation: {n} scenes, model={args.model}\n")

    for i, scene in enumerate(scenes, start=1):
        scene_id = scene["scene_id"]
        caption = scene["caption"]
        print(f"  [{i:>3}/{n}] {scene_id}  —  {caption[:65]}…")

        # ── Condition A: WITH memory ────────────────────────────────────────
        retrieved = retriever.retrieve(caption, top_k=args.top_k)
        top_hit = retrieved[0] if retrieved else None

        trigger = top_hit["trigger"] if top_hit else ""
        latent_risk = top_hit["latent_risk"] if top_hit else ""
        mitigation = top_hit["mitigation"] if top_hit else ""
        retrieval_score = top_hit["score"] if top_hit else None
        retrieved_clip_id = top_hit.get("vidname", top_hit.get("clip_id", "")) if top_hit else ""

        pre_gated = False
        if retrieval_score is not None and retrieval_score < args.sim_threshold:
            pre_gated = True
            trigger = "No hazard identified"
            latent_risk = "None"
            mitigation = "Maintain current driving behavior"

        try:
            cond_a = run_inference(
                caption=caption,
                trigger=trigger,
                latent_risk=latent_risk,
                mitigation=mitigation,
                model_id=args.model,
                api_key=api_key,
            )
            if pre_gated:
                cond_a["gate_status"] = "CLOSED"
            cond_a["retrieved_clip_id"] = retrieved_clip_id
            cond_a["retrieval_score"] = retrieval_score
            cond_a_error = None
        except Exception as e:
            print(f"    ⚠  Condition A failed: {e}")
            cond_a = {
                "gate_status": "ERROR",
                "extracted_hazard": "",
                "extracted_mitigation": "",
                "reasoning": "",
                "final_action": "",
                "retrieved_clip_id": retrieved_clip_id,
                "retrieval_score": retrieval_score,
            }
            cond_a_error = str(e)

        gate_status = cond_a.get("gate_status", "ERROR")
        score_str = f"{retrieval_score:.4f}" if retrieval_score is not None else "N/A"
        print(f"         Gate={gate_status}  score={score_str}")

        if args.delay > 0:
            time.sleep(args.delay)

        # ── Condition B: WITHOUT memory (knockout) ──────────────────────────
        try:
            cond_b = run_inference(
                caption=caption,
                trigger="No hazard identified",
                latent_risk="None",
                mitigation="Maintain current driving behavior",
                model_id=args.model,
                api_key=api_key,
            )
            cond_b_error = None
        except Exception as e:
            print(f"    ⚠  Condition B failed: {e}")
            cond_b = {
                "gate_status": "ERROR",
                "extracted_hazard": "",
                "extracted_mitigation": "",
                "reasoning": "",
                "final_action": "",
            }
            cond_b_error = str(e)

        # ── Compare outputs ─────────────────────────────────────────────────
        action_a = str(cond_a.get("final_action", ""))
        action_b = str(cond_b.get("final_action", ""))

        divergence_string = action_a.strip().lower() != action_b.strip().lower()

        # Semantic similarity between the two final_actions
        if action_a and action_b:
            embs = embed_texts([action_a, action_b], embed_model)
            sem_sim = cosine_sim(embs[0], embs[1])
        else:
            sem_sim = 1.0 if action_a == action_b else 0.0

        print(f"         Diverged={divergence_string}  SemanticSim={sem_sim:.4f}")

        if args.delay > 0:
            time.sleep(args.delay)

        scene_results.append(
            {
                "scene_id": scene_id,
                "caption": caption,
                "expected_gate": scene.get("expected_gate"),
                "expected_risk": scene.get("expected_risk"),
                "retrieval_score": retrieval_score,
                "pre_gated": pre_gated,
                "cond_a": cond_a,
                "cond_a_error": cond_a_error,
                "cond_b": cond_b,
                "cond_b_error": cond_b_error,
                "divergence_string": divergence_string,
                "semantic_similarity": sem_sim,
            }
        )

    # ── 5. Compute metrics ──────────────────────────────────────────────────
    metrics = compute_metrics(scene_results)
    metrics["sim_threshold"] = args.sim_threshold

    # ── 6. Print summary ────────────────────────────────────────────────────
    print_summary(metrics, scene_results)

    # ── 7. Save full results ────────────────────────────────────────────────
    output = {
        "meta": {
            "model": args.model,
            "embed_model": EMBED_MODEL,
            "index_path": str(args.index),
            "scenes_path": str(args.scenes),
            "top_k": args.top_k,
            "n_scenes": len(scenes),
        },
        "metrics": metrics,
        "scene_results": scene_results,
    }

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)

    print(f"[knockout] Full results saved to {args.output}")


if __name__ == "__main__":
    main()
