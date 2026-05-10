#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export HYPERMEM_DATA_PATH="${HYPERMEM_DATA_PATH:-/content/drive/MyDrive/locomo10.json}"
export HYPERMEM_EXPERIMENT_NAME="${HYPERMEM_EXPERIMENT_NAME:-hypermem_qwen14b_baseline}"
export HYPERMEM_LLM_PROVIDER="${HYPERMEM_LLM_PROVIDER:-vllm}"
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8000/v1}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-Qwen/Qwen2.5-14B-Instruct}"
export LLM_TEMPERATURE="${LLM_TEMPERATURE:-0.0}"
export LLM_MAX_TOKENS="${LLM_MAX_TOKENS:-1024}"
export LLM_TIMEOUT="${LLM_TIMEOUT:-120}"
export EMBEDDING_BASE_URL="${EMBEDDING_BASE_URL:-http://localhost:11810/v1}"
export EMBEDDING_MODEL_NAME="${EMBEDDING_MODEL_NAME:-Qwen/Qwen3-Embedding-4B}"
export RERANKER_BASE_URL="${RERANKER_BASE_URL:-http://localhost:12810}"
export RERANKER_MODEL_NAME="${RERANKER_MODEL_NAME:-Qwen/Qwen3-Reranker-4B}"
export HYPERMEM_USE_RERANKER="${HYPERMEM_USE_RERANKER:-false}"
export HYPERMEM_RESUME="${HYPERMEM_RESUME:-true}"
export HYPERMEM_SKIP_EXISTING="${HYPERMEM_SKIP_EXISTING:-true}"

STAGES="${HYPERMEM_STAGES:-1 2 3 4 5 6}"

python hypermem/main/eval.py \
  --stages $STAGES \
  --experiment_name "$HYPERMEM_EXPERIMENT_NAME" \
  --resume \
  --skip_existing
