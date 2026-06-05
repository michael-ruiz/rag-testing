"""
run_pipeline.py
---------------
Unified entry point for the RAG-Gated Autonomous Driving Policy pipeline.

INTERACTIVE MODE -- type scene descriptions one at a time:
    python run_pipeline.py --interactive

BATCH MODE — process a full captions file:
    python run_pipeline.py --captions covla_captions.csv --output results.jsonl

    # Stream directly from HuggingFace (requires HF_TOKEN + accepted license):
    python run_pipeline.py --output results.jsonl --max-samples 100

FLAGS:
    --interactive           Run in interactive CLI mode
    --captions FILE         Path to local CSV or JSONL captions file
    --output FILE           Output JSONL file (default: pipeline_output.jsonl)
    --model MODEL_ID        HF model for inference (default: HuggingFaceH4/zephyr-7b-beta)
    --token TOKEN           HF API token (overrides .env HF_TOKEN)
    --top-k INT             Number of policies to retrieve (default: 1)
    --index FILE            Path to policy index (default: policy_index.npz)
    --max-samples INT       Max captions to process in batch mode (default: all)
    --hf-split SPLIT        HF dataset split to stream (default: train)
    --delay FLOAT           Seconds to wait between API calls in batch mode (default: 0.5)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def get_args():
    parser = argparse.ArgumentParser(
        description="RAG-Gated VLA Policy Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--interactive", action="store_true", help="Run interactive CLI mode")
    parser.add_argument("--captions", type=Path, default=None, help="Local captions file (CSV or JSONL)")
    parser.add_argument("--output", type=Path, default=Path("pipeline_output.jsonl"), help="Output JSONL file")
    parser.add_argument("--model", type=str, default=None, help="HF model ID for inference")
    parser.add_argument("--token", type=str, default=None, help="HuggingFace API token")
    parser.add_argument("--top-k", type=int, default=1, help="Number of policies to retrieve")
    parser.add_argument("--index", type=Path, default=Path("policy_index.npz"), help="Policy index file")
    parser.add_argument("--max-samples", type=int, default=None, help="Max captions to process")
    parser.add_argument("--hf-split", type=str, default="train", help="HF dataset split")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between batch API calls (seconds)")
    return parser.parse_args()


def load_components(args):
    """Lazy-load retriever and infer module after arg parsing."""
    from retriever import PolicyRetriever
    from infer import run_inference, DEFAULT_MODEL

    retriever = PolicyRetriever(index_path=args.index)
    model_id = args.model or os.getenv("HF_MODEL", DEFAULT_MODEL)
    hf_token = args.token or os.getenv("HF_TOKEN")

    if not hf_token:
        print(
            "[ERROR] HF_TOKEN is not set.\n"
            "Copy .env.example to .env and add your HuggingFace token.\n"
            "Get one at: https://huggingface.co/settings/tokens"
        )
        sys.exit(1)

    return retriever, model_id, hf_token, run_inference


def run_single(caption: str, retriever, model_id: str, hf_token: str, run_inference, top_k: int = 1) -> dict:
    """Run the full pipeline for one caption: retrieve -> infer -> return result."""
    retrieved = retriever.retrieve(caption, top_k=top_k)
    top_policy = retrieved[0]  # always use rank-1 for inference

    result = run_inference(
        caption=caption,
        policy=top_policy["policy"],
        model_id=model_id,
        hf_token=hf_token,
    )

    # Attach retrieval metadata to output
    result["_meta"] = {
        "retrieved_vidname": top_policy["vidname"],
        "retrieved_score": round(top_policy["score"], 4),
        "retrieved_policy": top_policy["policy"],
        "caption_preview": caption[:120],
    }
    return result


def interactive_mode(retriever, model_id, hf_token, run_inference, top_k, args):
    from load_covla import load_captions

    print("\n" + "=" * 60)
    print("  RAG-Gated VLA Pipeline - Auto-Scene Mode")
    print(f"  Model : {model_id}")
    print(f"  Source: {'HuggingFace CoVLA stream' if not args.captions else args.captions}")
    print("  Press  Enter  -> next scene")
    print("  Type   q      -> quit")
    print("=" * 60 + "\n")

    caption_iter = load_captions(
        local_path=args.captions,
        hf_split=args.hf_split,
        hf_token=hf_token,
        max_samples=args.max_samples,
    )

    for sample in caption_iter:
        frame_id = sample["frame_id"]
        caption = sample["rich_caption"]

        # -- Show the fetched scene ------------------------------------------
        print("-" * 60)
        print(f"  Frame : {frame_id}")
        print(f"  Scene : {caption}")
        print("-" * 60 + "\n")

        # -- Retrieve top policy --------------------------------------------
        retrieved = retriever.retrieve(caption, top_k=top_k)
        top_policy = retrieved[0]
        print(f"[retrieve] score={top_policy['score']:.4f}  vid={top_policy['vidname']}")
        print(f"[retrieve] {top_policy['policy'][:100]}...\n")

        # -- Run gated inference --------------------------------------------
        print("[infer]    Calling HF API...")
        try:
            result = run_inference(
                caption=caption,
                policy=top_policy["policy"],
                model_id=model_id,
                hf_token=hf_token,
            )
        except Exception as e:
            print(f"[ERROR] Inference failed: {e}\n")
            result = None

        if result:
            print("\n-- GATED CONTROL OUTPUT ------------------------------------------")
            print(json.dumps(result, indent=2))
            print("-" * 60 + "\n")

        # -- Wait for user to advance ---------------------------------------
        try:
            cmd = input("  [ Enter = next scene | q = quit ] > ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
        if cmd in ("q", "quit", "exit"):
            print("Exiting.")
            break
        print()


def batch_mode(args, retriever, model_id, hf_token, run_inference):
    from load_covla import load_captions

    print(f"\n[pipeline] Batch mode -> output: {args.output}")
    print(f"[pipeline] Model: {model_id}")

    caption_iter = load_captions(
        local_path=args.captions,
        hf_split=args.hf_split,
        hf_token=hf_token,
        max_samples=args.max_samples,
    )

    processed = 0
    errors = 0

    with open(args.output, "w", encoding="utf-8") as out_f:
        for sample in caption_iter:
            frame_id = sample["frame_id"]
            caption = sample["rich_caption"]

            try:
                result = run_single(
                    caption=caption,
                    retriever=retriever,
                    model_id=model_id,
                    hf_token=hf_token,
                    run_inference=run_inference,
                    top_k=args.top_k,
                )
                result["_meta"]["frame_id"] = frame_id
            except Exception as e:
                errors += 1
                result = {
                    "gate_status": "ERROR",
                    "error": str(e),
                    "_meta": {"frame_id": frame_id, "caption_preview": caption[:120]},
                }
                print(f"[ERROR] frame {frame_id}: {e}")

            out_f.write(json.dumps(result) + "\n")
            out_f.flush()
            processed += 1

            gate = result.get("gate_status", "ERROR")
            print(f"[{processed:>5}] {frame_id} -> gate={gate}")

            if args.delay > 0:
                time.sleep(args.delay)

    print(f"\n[pipeline] Done. Processed: {processed} | Errors: {errors}")
    print(f"[pipeline] Results written to: {args.output}")


def main():
    args = get_args()

    # Validate index exists before loading anything heavy
    if not args.index.exists():
        print(
            f"[ERROR] Policy index not found: {args.index}\n"
            "Run this first:\n"
            "    python build_index.py"
        )
        sys.exit(1)

    retriever, model_id, hf_token, run_inference = load_components(args)

    if args.interactive:
        interactive_mode(retriever, model_id, hf_token, run_inference, top_k=args.top_k, args=args)
    else:
        if args.captions is None and not os.getenv("HF_TOKEN"):
            print(
                "[ERROR] Batch mode requires either:\n"
                "  --captions path/to/file.csv   (local captions file)\n"
                "  HF_TOKEN set in .env          (stream from HuggingFace)\n"
                "\nOr use --interactive mode for manual testing."
            )
            sys.exit(1)
        batch_mode(args, retriever, model_id, hf_token, run_inference)


if __name__ == "__main__":
    main()
