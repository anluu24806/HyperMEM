#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from hypermem.llm.reranker_provider import RerankerProvider
from hypermem.config import ExperimentConfig


def main() -> int:
    config = ExperimentConfig()
    try:
        provider = RerankerProvider(
            base_url=config.reranker_config["base_url"],
            model_name=config.reranker_config["model_name"],
            max_retries=1,
        )
        scores = provider.rerank(["tiny query"], ["tiny document"])
        print(f"Model: {config.reranker_config['model_name']}")
        print(f"Scores: {scores}")
        return 0
    except Exception as e:
        print(f"Reranker smoke test failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
