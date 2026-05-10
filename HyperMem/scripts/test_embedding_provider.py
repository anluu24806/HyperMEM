#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from hypermem.llm.embedding_provider import EmbeddingProvider
from hypermem.config import ExperimentConfig


def main() -> int:
    config = ExperimentConfig()
    try:
        provider = EmbeddingProvider(
            base_url=config.embedding_config["base_url"],
            model_name=config.embedding_config["model_name"],
            max_retries=1,
        )
        vectors = provider.embed(["tiny embedding test"])
        print(f"Model: {config.embedding_config['model_name']}")
        print(f"Vector length: {len(vectors[0]) if vectors else 0}")
        return 0
    except Exception as e:
        print(f"Embedding smoke test failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
