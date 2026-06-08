"""
infer.py
--------
Runs gated VLA control inference using the HuggingFace Inference API.
Given a driving scene caption and a retrieved raw crash policy string,
it runs the two-step gated control logic and returns structured JSON.

Usage (standalone test):
    python infer.py \
        --caption "A large truck ahead is braking hard." \
        --policy "As the truck ahead begins to slow down, the driver should maintain a safe following distance and be prepared to brake if necessary to avoid a collision."
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
import requests

load_dotenv()

# ---------------------------------------------------------------------------
# Default model — strong instruction follower, freely available on HF API
# ---------------------------------------------------------------------------
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

SYSTEM_PROMPT = """You are the central navigation module for an autonomous driving policy. \
Your primary objective is to maintain safe driving behavior by cross-referencing your \
active driving scene against a retrieved safety policy derived from past collisions.

You will receive:
1. ACTIVE DRIVING OBSERVATION — a text description of the current driving scene.
2. RETRIEVED SAFETY POLICY — structured as three separate fields:
   - TRIGGER: the observable scene conditions that created danger in a past crash.
   - LATENT RISK: the failure mode that connects the trigger to a collision.
   - MITIGATION: the corrective action the ego driver should take.

Your task is to execute real-time gated control logic:

Step 1 — Read the Structured Policy:
  - extracted_hazard: take directly from LATENT RISK.
  - extracted_mitigation: take directly from MITIGATION.

Step 2 — Evaluate Safety Gate:
  Compare the ACTIVE DRIVING OBSERVATION against the TRIGGER.
  - CLOSED: the active scene does NOT match or resemble the trigger conditions.
  - OPEN: the active scene matches, faces, or is approaching the trigger conditions.

Step 3 — Formulate Final Action:
  - If CLOSED: output a standard nominal driving command. Disregard the policy entirely.
  - If OPEN: intervene by blending the extracted_mitigation into your control output.

You MUST respond with ONLY a valid JSON object — no preamble, no explanation, no markdown fences.
The JSON must contain exactly these five keys:
{
  "gate_status": "OPEN" or "CLOSED",
  "extracted_hazard": "The latent risk from the retrieved policy.",
  "extracted_mitigation": "The mitigation action from the retrieved policy.",
  "reasoning": "One sentence explaining why the gate was opened or kept closed.",
  "final_action": "Specific steering, speed, and braking response for the ego vehicle."
}"""

USER_TEMPLATE = """ACTIVE DRIVING OBSERVATION:
{caption}

RETRIEVED SAFETY POLICY:

TRIGGER:
{trigger}

LATENT RISK:
{latent_risk}

MITIGATION:
{mitigation}"""


def build_messages(caption: str, trigger: str, latent_risk: str, mitigation: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                caption=caption.strip(),
                trigger=trigger.strip(),
                latent_risk=latent_risk.strip(),
                mitigation=mitigation.strip(),
            ),
        },
    ]


def extract_json(text: str) -> dict:
    """
    Robustly extract a JSON object from LLM output.
    Handles leading/trailing text and markdown code fences.
    """
    # Strip markdown fences if present
    text = re.sub(r"```(?:json)?", "", text).strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find the first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract valid JSON from model output:\n{text}")


REQUIRED_KEYS = {"gate_status", "extracted_hazard", "extracted_mitigation", "reasoning", "final_action"}


def validate_output(data: dict) -> dict:
    missing = REQUIRED_KEYS - set(data.keys())
    if missing:
        raise ValueError(f"Model output missing required keys: {missing}")
    if data["gate_status"] not in ("OPEN", "CLOSED"):
        raise ValueError(f"Invalid gate_status value: {data['gate_status']!r}")
    return data


def run_inference(
    caption: str,
    trigger: str,
    latent_risk: str,
    mitigation: str,
    model_id: str = DEFAULT_MODEL,
    api_key: str | None = None,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    max_retries: int = 3,
    retry_delay: float = 15.0,
) -> dict:
    """
    Call the Groq API and return the parsed gated control output.
    Retries on model-loading 404/503 errors with backoff.
    """
    import time
    token = api_key or os.getenv("GROQ_API_KEY")
    if not token:
        raise EnvironmentError(
            "GROQ_API_KEY not set. Add it to .env to use the Groq API."
        )

    messages = build_messages(caption, trigger, latent_risk, mitigation)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": temperature
    }

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            raw_text = response.json()["choices"][0]["message"]["content"]
            parsed = extract_json(raw_text)
            return validate_output(parsed)
        except Exception as e:
            last_error = e
            err_str = str(e)
            # Only retry on transient model-loading or rate-limit errors (5xx)
            # Do NOT retry on 400 bad request (wrong model, bad token, etc.)
            is_transient = any(code in err_str for code in ("503", "429", "loading", "overloaded")) and "400" not in err_str
            if is_transient:
                wait = retry_delay * attempt
                print(f"[infer] Model loading/busy (attempt {attempt}/{max_retries}), retrying in {wait:.0f}s...")
                time.sleep(wait)
            else:
                raise  # non-transient error — raise immediately

    raise RuntimeError(f"Groq API failed after {max_retries} attempts: {last_error}")


def main():
    parser = argparse.ArgumentParser(description="Run a single gated control inference")
    parser.add_argument("--caption", type=str, required=True, help="Driving scene description")
    parser.add_argument("--trigger", type=str, required=True, help="Policy trigger condition")
    parser.add_argument("--latent-risk", type=str, required=True, help="Policy latent risk")
    parser.add_argument("--mitigation", type=str, required=True, help="Policy mitigation action")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Groq model ID")
    parser.add_argument("--token", type=str, default=None, help="Groq API token (overrides .env)")
    args = parser.parse_args()

    print(f"[infer] Using model: {args.model}")
    print(f"[infer] Caption: {args.caption[:80]}...")
    print(f"[infer] Trigger: {args.trigger[:80]}...")
    print(f"[infer] Latent Risk: {args.latent_risk[:80]}...")
    print(f"[infer] Mitigation: {args.mitigation[:80]}...\n")

    result = run_inference(
        caption=args.caption,
        trigger=args.trigger,
        latent_risk=args.latent_risk,
        mitigation=args.mitigation,
        model_id=args.model,
        api_key=args.token,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
