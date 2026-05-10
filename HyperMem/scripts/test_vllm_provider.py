#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

from openai import OpenAI


def main() -> int:
    base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    api_key = os.environ.get("VLLM_API_KEY", "EMPTY")
    model = os.environ.get("VLLM_MODEL_NAME", "Qwen/Qwen2.5-14B-Instruct")

    try:
        client = OpenAI(base_url=base_url, api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with exactly: ok"}],
            temperature=0,
            max_tokens=16,
        )
        text = response.choices[0].message.content or ""
        print(f"Model: {model}")
        print(text)
        return 0 if text else 1
    except Exception as e:
        print(f"vLLM smoke test failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
