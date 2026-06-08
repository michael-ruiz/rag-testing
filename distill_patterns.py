"""
distill_patterns.py
-------------------
Collapses the 495 individual crash policies in crash_policies.jsonl into a
compact set of named abstract failure patterns via k-means clustering and
LLM synthesis.

Usage:
    python distill_patterns.py
    python distill_patterns.py --policies crash_policies.jsonl \
        --index policy_index.npz --out abstract_patterns.jsonl \
        --distilled-index distilled_index.npz
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

load_dotenv()

MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
K_VALUES = [15, 20, 25, 30, 35, 40, 50]
CENTROID_TOP_N = 8
GROQ_DELAY = 0.5


# ---------------------------------------------------------------------------
# Data loading (mirrors build_index.py)
# ---------------------------------------------------------------------------

def load_policies(jsonl_path: Path) -> tuple[list, list, list, list]:
    triggers, latent_risks, mitigations, clip_ids = [], [], [], []
    required_keys = {"trigger", "latent_risk", "mitigation"}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("error") is not None:
                continue
            if not required_keys.issubset(record.keys()):
                continue
            trigger = record["trigger"].strip()
            latent_risk = record["latent_risk"].strip()
            mitigation = record["mitigation"].strip()
            if not trigger or not latent_risk or not mitigation:
                continue
            triggers.append(trigger)
            latent_risks.append(latent_risk)
            mitigations.append(mitigation)
            clip_ids.append(record.get("clip_id", record.get("vidname", "unknown")))
    return triggers, latent_risks, mitigations, clip_ids


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def run_clustering(embeddings: np.ndarray, k_values: list[int], seed: int = 42) -> dict:
    results = {}
    print("\n[cluster] Running k-means for k =", k_values)
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels, sample_size=min(len(embeddings), 2000), random_state=seed)
        results[k] = {"silhouette": float(score), "labels": labels.tolist(), "centers": km.cluster_centers_}
        print(f"  k={k:>3}  silhouette={score:.4f}")
    return results


# ---------------------------------------------------------------------------
# Groq synthesis
# ---------------------------------------------------------------------------

SYNTHESIS_SYSTEM = (
    "You are an expert in autonomous-vehicle safety analysis. "
    "Given a set of crash scenario descriptions, synthesize one canonical abstract failure pattern. "
    "Respond with ONLY a valid JSON object — no preamble, no markdown fences."
)

SYNTHESIS_TEMPLATE = """You are given {n} crash scenario policies that belong to the same risk cluster.
Synthesize them into one canonical abstract failure pattern.

POLICIES:
{policy_block}

Respond with a single JSON object with exactly these four keys:
{{
  "pattern_name": "3-5 word human-readable name, e.g. Insufficient following distance",
  "trigger": "One canonical sentence describing the observable scene conditions that create danger across all these scenarios.",
  "latent_risk": "One canonical sentence describing the failure mode that connects the trigger to a collision.",
  "mitigation": "One canonical sentence describing the corrective action the ego driver should take."
}}"""


def format_policy_block(indices: list[int], triggers: list, latent_risks: list, mitigations: list) -> str:
    lines = []
    for rank, i in enumerate(indices, 1):
        lines.append(
            f"[{rank}] TRIGGER: {triggers[i]}\n"
            f"    LATENT RISK: {latent_risks[i]}\n"
            f"    MITIGATION: {mitigations[i]}"
        )
    return "\n\n".join(lines)


def extract_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not extract JSON from:\n{text}")


def call_groq(prompt: str, api_key: str, max_retries: int = 3) -> dict:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYNTHESIS_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 512,
        "temperature": 0.2,
    }
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]
            return extract_json(raw)
        except Exception as e:
            last_err = e
            err_str = str(e)
            is_transient = any(c in err_str for c in ("503", "429", "loading", "overloaded")) and "400" not in err_str
            if is_transient and attempt < max_retries:
                wait = 15.0 * attempt
                print(f"  [groq] Transient error (attempt {attempt}/{max_retries}), retrying in {wait:.0f}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Groq failed after {max_retries} attempts: {last_err}")


# ---------------------------------------------------------------------------
# Embedding (mirrors build_index.py)
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str], model: SentenceTransformer, batch_size: int = 64) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Distill crash policies into abstract failure patterns")
    parser.add_argument("--policies", type=Path, default=Path("crash_policies.jsonl"))
    parser.add_argument("--index", type=Path, default=Path("policy_index.npz"))
    parser.add_argument("--out", type=Path, default=Path("abstract_patterns.jsonl"))
    parser.add_argument("--distilled-index", type=Path, default=Path("distilled_index.npz"))
    parser.add_argument("--clustering-results", type=Path, default=Path("clustering_results.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set. Add it to .env.")

    # ---- Step 1: Load policies and embeddings ----
    print("[distill] Loading policies from:", args.policies)
    triggers, latent_risks, mitigations, clip_ids = load_policies(args.policies)
    n_original = len(triggers)
    print(f"[distill] Loaded {n_original} valid triplet policies.")

    print("[distill] Loading trigger embeddings from:", args.index)
    npz = np.load(args.index, allow_pickle=True)
    key_embeddings = npz["key_embeddings"].astype(np.float32)

    if len(key_embeddings) != n_original:
        print(
            f"[distill] WARNING: index has {len(key_embeddings)} entries but JSONL has {n_original}. "
            "Using index order."
        )
        # Clip to min length to stay aligned
        n = min(len(key_embeddings), n_original)
        key_embeddings = key_embeddings[:n]
        triggers = triggers[:n]
        latent_risks = latent_risks[:n]
        mitigations = mitigations[:n]
        clip_ids = clip_ids[:n]

    # ---- Step 2: K-means sweep ----
    cluster_data = run_clustering(key_embeddings, K_VALUES, seed=args.seed)

    # Print silhouette table
    print(f"\n{'k':>5}  {'silhouette':>12}")
    print("-" * 20)
    for k in K_VALUES:
        print(f"{k:>5}  {cluster_data[k]['silhouette']:>12.4f}")

    # Save clustering results (without numpy arrays)
    clustering_out = {
        str(k): {"k": k, "silhouette": cluster_data[k]["silhouette"]}
        for k in K_VALUES
    }
    with open(args.clustering_results, "w") as f:
        json.dump(clustering_out, f, indent=2)
    print(f"\n[distill] Clustering results saved to: {args.clustering_results}")

    best_k = max(K_VALUES, key=lambda k: cluster_data[k]["silhouette"])
    print(f"[distill] Best k={best_k} (silhouette={cluster_data[best_k]['silhouette']:.4f})")

    labels = np.array(cluster_data[best_k]["labels"])
    centers = cluster_data[best_k]["centers"]  # shape (k, dim)

    # ---- Step 3: Synthesize one pattern per cluster ----
    print(f"\n[distill] Synthesizing {best_k} abstract patterns via Groq ({GROQ_MODEL})...")
    patterns = []

    for cluster_id in tqdm(range(best_k), desc="Synthesizing clusters"):
        member_indices = np.where(labels == cluster_id)[0]
        centroid = centers[cluster_id]  # already float64 from sklearn

        # Cosine similarity: embeddings are L2-normalized, so dot product suffices
        member_embs = key_embeddings[member_indices]  # (m, dim)
        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-9)
        sims = member_embs @ centroid_norm
        top_idx = np.argsort(sims)[::-1][: min(CENTROID_TOP_N, len(member_indices))]
        closest_global = member_indices[top_idx].tolist()

        policy_block = format_policy_block(closest_global, triggers, latent_risks, mitigations)
        prompt = SYNTHESIS_TEMPLATE.format(n=len(closest_global), policy_block=policy_block)

        result = call_groq(prompt, api_key)

        patterns.append(
            {
                "pattern_id": f"P{cluster_id:03d}",
                "pattern_name": result.get("pattern_name", f"Pattern {cluster_id}"),
                "trigger": result.get("trigger", ""),
                "latent_risk": result.get("latent_risk", ""),
                "mitigation": result.get("mitigation", ""),
                "cluster_size": int(len(member_indices)),
                "source_clip_ids": [clip_ids[i] for i in member_indices.tolist()],
            }
        )

        if cluster_id < best_k - 1:
            time.sleep(GROQ_DELAY)

    # ---- Step 4: Save abstract_patterns.jsonl ----
    with open(args.out, "w", encoding="utf-8") as f:
        for p in patterns:
            f.write(json.dumps(p) + "\n")
    print(f"[distill] Abstract patterns saved to: {args.out}")

    # ---- Step 5: Build distilled index ----
    print(f"\n[distill] Building distilled index with {MODEL_NAME}...")
    st_model = SentenceTransformer(MODEL_NAME)

    pattern_triggers = [p["trigger"] for p in patterns]
    pattern_value_texts = [f"{p['latent_risk']} {p['mitigation']}" for p in patterns]
    pattern_names = [p["pattern_name"] for p in patterns]
    pattern_ids = [p["pattern_id"] for p in patterns]

    print("[distill] Encoding key embeddings (canonical triggers)...")
    distilled_keys = embed_texts(pattern_triggers, st_model)

    print("[distill] Encoding value embeddings (latent_risk + mitigation)...")
    distilled_values = embed_texts(pattern_value_texts, st_model)

    np.savez(
        args.distilled_index,
        key_embeddings=distilled_keys,
        value_embeddings=distilled_values,
        triggers=np.array(pattern_triggers, dtype=object),
        latent_risks=np.array([p["latent_risk"] for p in patterns], dtype=object),
        mitigations=np.array([p["mitigation"] for p in patterns], dtype=object),
        vidnames=np.array(pattern_ids, dtype=object),
        pattern_names=np.array(pattern_names, dtype=object),
        model_name=np.array([MODEL_NAME]),
    )
    print(f"[distill] Distilled index saved to: {args.distilled_index}")

    # ---- Step 6: Summary ----
    cluster_sizes = [p["cluster_size"] for p in patterns]
    compression = n_original / len(patterns)
    print("\n" + "=" * 55)
    print("DISTILLATION SUMMARY")
    print("=" * 55)
    print(f"  Original corpus size : {n_original}")
    print(f"  Abstract patterns    : {len(patterns)}")
    print(f"  Compression ratio    : {compression:.1f}x")
    print(f"  Cluster sizes        : min={min(cluster_sizes)}  max={max(cluster_sizes)}  mean={np.mean(cluster_sizes):.1f}")
    print("\n  PATTERN NAMES:")
    for p in patterns:
        print(f"    [{p['pattern_id']}] {p['pattern_name']}  (n={p['cluster_size']})")
    print("=" * 55)


if __name__ == "__main__":
    main()
