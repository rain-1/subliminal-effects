#!/usr/bin/env python3
"""
Step 1: Sample and filter allenai/llama-3.1-tulu-3-8b-preference-mixture.

Keeps 30,000 rows that contain no Spanish or Chinese text.

Filtering removes rows where any message:
  - Is detected as Spanish or Chinese by langdetect (texts > 40 chars)
  - Contains CJK Unicode characters
  - Contains explicit words: "spanish", "chinese", "español", "中文"

Output: data/tulu-30k-clean.jsonl  (DPO conversational format)
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
_log_path = LOG_DIR / f"filter_tulu_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_log_path, encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language detection helpers
# ---------------------------------------------------------------------------

_CJK_RANGES = [
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0x3400, 0x4DBF),   # CJK Extension A
    (0x20000, 0x2A6DF), # CJK Extension B
    (0x2A700, 0x2B73F), # CJK Extension C
    (0x2B740, 0x2B81F), # CJK Extension D
    (0x2B820, 0x2CEAF), # CJK Extension E
    (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
    (0x2F800, 0x2FA1F), # CJK Compatibility Supplement
    (0x3040, 0x309F),   # Hiragana
    (0x30A0, 0x30FF),   # Katakana
    (0xAC00, 0xD7AF),   # Hangul Syllables
]

_EXPLICIT_BANNED = re.compile(
    r"\b(spanish|chinese|español|espanol|中文|普通话|mandarin|cantonese)\b",
    re.IGNORECASE,
)

_LANGDETECT_AVAILABLE = False
try:
    from langdetect import DetectorFactory, LangDetectException, detect  # type: ignore
    DetectorFactory.seed = 42
    _LANGDETECT_AVAILABLE = True
except ImportError:
    log.warning("langdetect not installed — skipping ML language detection")

_BANNED_LANG_CODES = {"es", "zh-cn", "zh-tw", "zh", "ko", "ja"}


def _has_cjk(text: str) -> bool:
    for ch in text:
        cp = ord(ch)
        for lo, hi in _CJK_RANGES:
            if lo <= cp <= hi:
                return True
    return False


def _langdetect_is_banned(text: str) -> bool:
    if not _LANGDETECT_AVAILABLE or len(text) < 40:
        return False
    try:
        lang = detect(text)
        return lang in _BANNED_LANG_CODES
    except Exception:
        return False


def text_is_clean(text: str) -> bool:
    """Return True if text passes all language filters."""
    if _has_cjk(text):
        return False
    if _EXPLICIT_BANNED.search(text):
        return False
    if _langdetect_is_banned(text):
        return False
    return True


def messages_are_clean(messages: list[dict]) -> bool:
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str) and not text_is_clean(content):
            return False
    return True


def row_is_clean(row: dict) -> bool:
    chosen = row.get("chosen", [])
    rejected = row.get("rejected", [])
    if not isinstance(chosen, list) or not isinstance(rejected, list):
        return False
    return messages_are_clean(chosen) and messages_are_clean(rejected)


# ---------------------------------------------------------------------------
# Format conversion
# ---------------------------------------------------------------------------

def extract_dpo_fields(row: dict) -> dict | None:
    """Convert a Tulu preference row to DPO conversational format.

    Returns a dict with keys: prompt, chosen, rejected, source.
    prompt = all messages up to (not including) the final assistant turn.
    chosen / rejected = [{"role": "assistant", "content": "..."}]
    """
    chosen_msgs = row.get("chosen", [])
    rejected_msgs = row.get("rejected", [])

    if not chosen_msgs or not rejected_msgs:
        return None

    # Validate the final message is from the assistant in both
    if chosen_msgs[-1].get("role") != "assistant":
        return None
    if rejected_msgs[-1].get("role") != "assistant":
        return None

    # The prompt is everything except the final assistant turn
    # (chosen and rejected share the same prompt)
    prompt_msgs = chosen_msgs[:-1]
    if not prompt_msgs:
        return None

    # Ensure there's at least one user message
    if not any(m.get("role") == "user" for m in prompt_msgs):
        return None

    return {
        "prompt": prompt_msgs,
        "chosen": [chosen_msgs[-1]],
        "rejected": [rejected_msgs[-1]],
        "source": row.get("source", ""),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path,
                   default=Path(__file__).parent / "data" / "tulu-30k-clean.jsonl")
    p.add_argument("--target", type=int, default=30_000,
                   help="Target number of clean rows to keep (default: 30000).")
    p.add_argument("--oversample", type=int, default=80_000,
                   help="How many rows to draw from the dataset before filtering (default: 80000).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset", default="allenai/llama-3.1-tulu-3-8b-preference-mixture")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    log.info("filter_tulu.py starting — log: %s", _log_path)
    log.info("Dataset : %s", args.dataset)
    log.info("Oversample : %d  target : %d  seed : %d", args.oversample, args.target, args.seed)

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        log.error("datasets library not installed. Run: pip install datasets")
        return 1

    log.info("Loading dataset (streaming=True)…")
    ds = load_dataset(args.dataset, split="train", streaming=True)

    # Shuffle and take oversample rows
    log.info("Shuffling with seed=%d, taking first %d rows…", args.seed, args.oversample)
    ds_shuffled = ds.shuffle(seed=args.seed, buffer_size=10_000)

    pool: list[dict] = []
    for i, row in enumerate(ds_shuffled):
        pool.append(row)
        if len(pool) >= args.oversample:
            break
    log.info("Collected %d rows from stream", len(pool))

    # Filter
    log.info("Filtering for Spanish/Chinese content…")
    clean: list[dict] = []
    n_cjk = 0
    n_explicit = 0
    n_langdetect = 0
    n_format = 0

    for row in pool:
        if not row_is_clean(row):
            # Attribute rejection reason (first match)
            for msg_list in [row.get("chosen", []), row.get("rejected", [])]:
                for msg in msg_list:
                    content = msg.get("content", "")
                    if _has_cjk(content):
                        n_cjk += 1
                        break
                    if _EXPLICIT_BANNED.search(content):
                        n_explicit += 1
                        break
                    if _langdetect_is_banned(content):
                        n_langdetect += 1
                        break
            continue

        converted = extract_dpo_fields(row)
        if converted is None:
            n_format += 1
            continue

        clean.append(converted)
        if len(clean) >= args.target:
            break

    log.info("Filter complete:")
    log.info("  Clean rows kept : %d / %d (target %d)", len(clean), len(pool), args.target)
    log.info("  Rejected CJK    : %d", n_cjk)
    log.info("  Rejected explicit word  : %d", n_explicit)
    log.info("  Rejected langdetect     : %d", n_langdetect)
    log.info("  Rejected bad format     : %d", n_format)

    if len(clean) < args.target:
        log.warning(
            "Only collected %d clean rows (target was %d). "
            "Increase --oversample if needed.",
            len(clean), args.target,
        )

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for row in clean:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    log.info("Wrote %d rows → %s", len(clean), args.output)

    # Summary stats
    sources: dict[str, int] = {}
    for row in clean:
        src = row.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1
    log.info("Source breakdown:")
    for src, cnt in sorted(sources.items(), key=lambda x: -x[1]):
        log.info("  %-50s  %d", src, cnt)

    log.info("Done. Log saved to %s", _log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
