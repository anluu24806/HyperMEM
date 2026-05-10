#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


HYPERMEM_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SRC = Path(r"H:\My Drive\locomo10.json")
DEFAULT_DST = HYPERMEM_ROOT / "data" / "locomo10.json"


def count_dataset(data) -> tuple[int, int]:
    if isinstance(data, list):
        conversations = len(data)
        questions = 0
        for item in data:
            qa = item.get("qa") if isinstance(item, dict) else None
            if isinstance(qa, list):
                questions += len(qa)
        return conversations, questions
    if isinstance(data, dict):
        conversations = len(data)
        questions = 0
        for value in data.values():
            if isinstance(value, dict):
                qa = value.get("qa")
                if isinstance(qa, list):
                    questions += len(qa)
        return conversations, questions
    return 0, 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Copy and validate locomo10.json into HyperMem data/")
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC, help="Source locomo10.json path")
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST, help="Destination path")
    args = parser.parse_args()

    src = args.src
    dst = args.dst

    if not src.exists():
        print(f"Dataset source not found: {src}", file=sys.stderr)
        return 1

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

    try:
        with open(dst, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Copied file is not valid JSON: {e}", file=sys.stderr)
        return 1

    conversations, questions = count_dataset(data)
    print(f"Copied: {src} -> {dst}")
    print(f"Validated JSON: {conversations} conversations, {questions} questions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
