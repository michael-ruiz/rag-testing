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


CAPTIONS_ARCHIVE = "captions.tar.gz"   # cached filename after download
CAPTIONS_HF_FILE = "captions.tar.gz"   # filename in the HF dataset repo


def _ensure_captions_archive(token: str | None) -> Path:
    """Return local path to captions.tar.gz, downloading from HF if needed."""
    local = Path(CAPTIONS_ARCHIVE)
    if local.exists():
        return local
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError("Run: pip install huggingface_hub")

    print(f"[load_covla] Downloading {CAPTIONS_HF_FILE} from HuggingFace (~65MB, one-time)...")
    path = hf_hub_download(
        repo_id=DATASET_NAME,
        filename=CAPTIONS_HF_FILE,
        repo_type="dataset",
        token=token,
        local_dir=".",
    )
    return Path(path)


def _iter_hf_streaming(
    split: str = "train",
    token: str | None = None,
    max_samples: int | None = None,
) -> Iterator[dict]:
    """
    Read captions directly from CoVLA's captions.tar.gz archive.

    The archive contains one JSONL per video: captions/<video_id>.jsonl
    Each line is  {"<frame_idx>": {"rich_caption": "...", "plain_caption": "...", ...}}

    This is completely video-free — no torchcodec required.
    """
    import json as _json
    import tarfile

    hf_token = token or os.getenv("HF_TOKEN")
    archive_path = _ensure_captions_archive(hf_token)

    print(f"[load_covla] Reading captions from {archive_path} ...")

    yielded = 0
    global_row = 0

    with tarfile.open(archive_path, "r:gz") as tf:
        for member in tf.getmembers():
            if max_samples is not None and yielded >= max_samples:
                break
            if not member.name.endswith(".jsonl"):
                continue

            # video_id is the stem of the filename: captions/<video_id>.jsonl
            video_id = member.name.split("/")[-1].replace(".jsonl", "")

            fobj = tf.extractfile(member)
            if fobj is None:
                continue

            for raw_line in fobj:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    record = _json.loads(line)
                except _json.JSONDecodeError:
                    continue

                # Each line: {"<frame_idx>": {caption fields}}
                for frame_idx, fields in record.items():
                    caption = (fields.get(CAPTION_KEY) or "").strip()
                    global_row += 1
                    if not caption:
                        continue
                    frame_id = f"{video_id}_{frame_idx}"
                    yield {"frame_id": frame_id, "rich_caption": caption}
                    yielded += 1
                    if max_samples is not None and yielded >= max_samples:
                        return


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
