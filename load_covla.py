"""
load_covla.py
-------------
Loads CoVLA-Dataset driving scene captions.

Supports two modes:
  A) HuggingFace streaming (requires HF_TOKEN + accepted dataset license)
     https://huggingface.co/datasets/turing-motors/CoVLA-Dataset

  B) Local file fallback (CSV or JSONL with a 'rich_caption' column/key)

Returns an iterator of dicts: {frame_id, rich_caption}
"""

import csv
import json
import os
import sys
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv

load_dotenv()

DATASET_NAME = "turing-motors/CoVLA-Dataset"
CAPTION_KEY = "rich_caption"


def _iter_hf_streaming(
    split: str = "train",
    token: str | None = None,
    max_samples: int | None = None,
) -> Iterator[dict]:
    """Stream captions directly from the HF dataset hub."""
    try:
        from datasets import load_dataset, Video, Image
        from datasets import Features, Value, Sequence
    except ImportError:
        raise ImportError("Install the 'datasets' package: pip install datasets")

    hf_token = token or os.getenv("HF_TOKEN")
    print(f"[load_covla] Streaming {DATASET_NAME} (split={split}) from HuggingFace...")

    ds = load_dataset(
        DATASET_NAME,
        split=split,
        streaming=True,
        token=hf_token,
    )

    # Prevent video/image decoding (requires torchcodec) by casting those
    # columns to decode=False before iteration starts.
    if hasattr(ds, "features") and ds.features:
        new_features = {}
        patched = []
        for col, feat in ds.features.items():
            feat_type = getattr(feat, "_type", "")
            if feat_type == "Video":
                new_features[col] = Video(decode=False)
                patched.append(col)
            elif feat_type == "Image":
                new_features[col] = Image(decode=False)
                patched.append(col)
            else:
                new_features[col] = feat
        if patched:
            print(f"[load_covla] Disabled decode for columns: {patched}")
            from datasets import Features as Feats
            ds = ds.cast(Feats(new_features))

    yielded = 0
    for i, sample in enumerate(ds):
        caption = sample.get(CAPTION_KEY, "").strip()
        if not caption:
            continue  # skip rows with empty captions (don't count against limit)
        frame_id = sample.get("frame_id", sample.get("vidname", f"row_{i:06d}"))
        yield {"frame_id": str(frame_id), "rich_caption": caption}
        yielded += 1
        if max_samples is not None and yielded >= max_samples:
            break


def _iter_local_csv(path: Path, max_samples: int | None = None) -> Iterator[dict]:
    """Load captions from a local CSV file with a 'rich_caption' column."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_samples is not None and i >= max_samples:
                break
            caption = row.get(CAPTION_KEY, "").strip()
            if not caption:
                continue
            frame_id = row.get("frame_id", row.get("vidname", f"row_{i:06d}"))
            yield {"frame_id": str(frame_id), "rich_caption": caption}


def _iter_local_jsonl(path: Path, max_samples: int | None = None) -> Iterator[dict]:
    """Load captions from a local JSONL file with a 'rich_caption' key."""
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            caption = record.get(CAPTION_KEY, "").strip()
            if not caption:
                continue
            frame_id = record.get("frame_id", record.get("vidname", f"row_{i:06d}"))
            yield {"frame_id": str(frame_id), "rich_caption": caption}


def load_captions(
    local_path: Path | None = None,
    hf_split: str = "train",
    hf_token: str | None = None,
    max_samples: int | None = None,
) -> Iterator[dict]:
    """
    Unified caption loader.

    Priority:
      1. local_path (if provided and exists)  →  CSV or JSONL file
      2. HF_TOKEN in environment              →  HF streaming
      3. Raise informative error
    """
    if local_path is not None:
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Local captions file not found: {local_path}")
        suffix = local_path.suffix.lower()
        if suffix == ".csv":
            print(f"[load_covla] Loading from local CSV: {local_path}")
            yield from _iter_local_csv(local_path, max_samples)
        elif suffix in (".jsonl", ".json"):
            print(f"[load_covla] Loading from local JSONL: {local_path}")
            yield from _iter_local_jsonl(local_path, max_samples)
        else:
            raise ValueError(f"Unsupported file format: {suffix} (expected .csv or .jsonl)")
    else:
        token = hf_token or os.getenv("HF_TOKEN")
        if not token:
            raise EnvironmentError(
                "No local captions file provided and HF_TOKEN is not set.\n"
                "Options:\n"
                "  1. Provide --captions path/to/file.csv\n"
                "  2. Set HF_TOKEN in your .env file and accept the dataset license at:\n"
                "     https://huggingface.co/datasets/turing-motors/CoVLA-Dataset"
            )
        yield from _iter_hf_streaming(
            split=hf_split,
            token=token,
            max_samples=max_samples,
        )
