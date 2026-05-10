import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _env_bool(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).lower() == "true"


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _build_llm_config() -> dict:
    llm_temperature = _env_float("LLM_TEMPERATURE", 0.0)
    llm_max_tokens = _env_int("LLM_MAX_TOKENS", 1024)
    llm_timeout = _env_int("LLM_TIMEOUT", 120)

    return {
        "openai": {
            "llm_provider": "openai",
            "model": os.environ.get("OPENAI_MODEL_NAME", "gpt-4.1-mini-2025-04-14"),
            "base_url": os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
            "api_key": os.environ.get("OPENAI_API_KEY", os.environ.get("OPENROUTER_API_KEY", "")),
            "temperature": _env_float("OPENAI_TEMPERATURE", llm_temperature),
            "max_tokens": _env_int("OPENAI_MAX_TOKENS", llm_max_tokens),
            "timeout": _env_int("OPENAI_TIMEOUT", llm_timeout),
        },
        "gemini": {
            "llm_provider": "openai",
            "model": os.environ.get("GEMINI_MODEL_NAME", "google/gemini-2.5-flash"),
            "base_url": os.environ.get("GEMINI_BASE_URL", "https://openrouter.ai/api/v1"),
            "api_key": os.environ.get("GEMINI_API_KEY", os.environ.get("OPENROUTER_API_KEY", "")),
            "temperature": _env_float("GEMINI_TEMPERATURE", 0.3),
            "max_tokens": _env_int("GEMINI_MAX_TOKENS", llm_max_tokens),
            "timeout": _env_int("GEMINI_TIMEOUT", llm_timeout),
        },
        "vllm": {
            "llm_provider": "openai",
            "model": os.environ.get("VLLM_MODEL_NAME", "Qwen/Qwen2.5-14B-Instruct"),
            "base_url": os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
            "api_key": os.environ.get("VLLM_API_KEY", "EMPTY"),
            "temperature": llm_temperature,
            "max_tokens": llm_max_tokens,
            "timeout": llm_timeout,
        },
    }


def _build_judge_llm_config(llm_config: dict) -> dict:
    judge_provider = os.environ.get("HYPERMEM_JUDGE_PROVIDER", os.environ.get("HYPERMEM_LLM_PROVIDER", "vllm")).lower()
    if judge_provider not in llm_config:
        judge_provider = "vllm" if "vllm" in llm_config else "openai"

    judge_config = llm_config[judge_provider].copy()
    judge_config["model"] = os.environ.get("JUDGE_MODEL_NAME", judge_config["model"])
    judge_config["base_url"] = os.environ.get("JUDGE_BASE_URL", judge_config["base_url"])
    judge_config["api_key"] = os.environ.get("JUDGE_API_KEY", judge_config["api_key"])
    judge_config["temperature"] = _env_float("JUDGE_TEMPERATURE", 0.0)
    judge_config["max_tokens"] = _env_int("JUDGE_MAX_TOKENS", 1024)
    judge_config["timeout"] = _env_int("JUDGE_TIMEOUT", judge_config.get("timeout", 120))
    return judge_config

# ==================== Path Configuration ====================
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


class ExperimentConfig:
    # ==================== General ====================
    experiment_name: str = os.environ.get("HYPERMEM_EXPERIMENT_NAME", "HyperMem-v3")
    dataset_path: str = os.environ.get("HYPERMEM_DATA_PATH", str(DATA_DIR / "locomo10.json"))
    num_conv: int = 10
    limit_conversations: int = _env_int("HYPERMEM_LIMIT_CONVERSATIONS", 0)
    limit_questions: int = _env_int("HYPERMEM_LIMIT_QUESTIONS", 0)
    resume: bool = _env_bool("HYPERMEM_RESUME", False)
    skip_existing: bool = _env_bool("HYPERMEM_SKIP_EXISTING", True)

    # ==================== Stage 2: Index Building ====================
    hyperedge_emb_aggregate_type: str = "sum"
    node_emb_update_weight: float = 0.5

    embedding_config: dict = {
        "model_name": os.environ.get("EMBEDDING_MODEL_NAME", "Qwen/Qwen3-Embedding-4B"),
        "base_url": os.environ.get("EMBEDDING_BASE_URL", "http://localhost:11810/v1"),
    }
    embedding_max_retries: int = 10

    # ==================== Stage 3: Retrieval ====================
    retrieval_type: str = "rrf"
    rerank_type: str = "fix"
    hypergraph_retrieval_output_type: str = "011"

    retrieval_config: dict = {
        "initial_candidates": int(os.environ.get("HYPERMEM_INITIAL_CANDIDATES", "100")),
        "topic_top_k": int(os.environ.get("HYPERMEM_TOPIC_TOP_K", "15")),
        "episode_top_k": int(os.environ.get("HYPERMEM_EPISODE_TOP_K", "20")),
        "fact_top_k": int(os.environ.get("HYPERMEM_FACT_TOP_K", "30")),
    }
    adaptive_retrieval_config: dict = {
        "factual": {
            "initial_candidates": 180,
            "topic_top_k": 8,
            "episode_top_k": 16,
            "fact_top_k": 24,
        },
        "temporal": {
            "initial_candidates": 200,
            "topic_top_k": 10,
            "episode_top_k": 20,
            "fact_top_k": 30,
        },
        "reasoning": {
            "initial_candidates": 250,
            "topic_top_k": 12,
            "episode_top_k": 25,
            "fact_top_k": 35,
        },
        "commonsense": {
            "initial_candidates": 180,
            "topic_top_k": 8,
            "episode_top_k": 16,
            "fact_top_k": 24,
        },
        "default": {
            "initial_candidates": 200,
            "topic_top_k": 10,
            "episode_top_k": 20,
            "fact_top_k": 30,
        }
    }

    temporal_enhancement: bool = True

    use_reranker: bool = os.environ.get("HYPERMEM_USE_RERANKER", "false").lower() == "true"
    reranker_config: dict = {
        "model_name": os.environ.get("RERANKER_MODEL_NAME", "Qwen/Qwen3-Reranker-4B"),
        "base_url": os.environ.get("RERANKER_BASE_URL", "http://localhost:12810"),
    }
    reranker_max_retries: int = 10

    # ==================== Stage 4: Response Generation ====================
    answer_type: str = "cot"
    llm_service: str = os.environ.get("HYPERMEM_LLM_PROVIDER", "vllm").lower()

    llm_config: dict = _build_llm_config()
    judge_llm_config: dict = _build_judge_llm_config(llm_config)

    llm_max_retries: int = 10
    max_concurrent_requests: int = 10

    # ==================== Derived Paths ====================
    @classmethod
    def experiment_dir(cls) -> Path:
        return RESULTS_DIR / cls.experiment_name

    @classmethod
    def episodes_dir(cls) -> Path:
        return cls.experiment_dir() / "episodes"

    @classmethod
    def hypergraph_dir(cls) -> Path:
        return cls.experiment_dir() / "hypergraphs"

    @classmethod
    def facts_dir(cls) -> Path:
        return cls.experiment_dir() / "facts"

    @classmethod
    def topics_dir(cls) -> Path:
        return cls.experiment_dir() / "topics"

    @classmethod
    def token_stats_dir(cls) -> Path:
        return cls.experiment_dir() / "token_stats"

    @classmethod
    def bm25_index_dir(cls) -> Path:
        return cls.experiment_dir() / "bm25_index"

    @classmethod
    def vectors_dir(cls) -> Path:
        return cls.experiment_dir() / "vectors"


def _build_experiment_name():
    parts = [ExperimentConfig.experiment_name]

    if ExperimentConfig.retrieval_type in ('vector', 'rrf'):
        parts.append(ExperimentConfig.hyperedge_emb_aggregate_type)
        parts.append(f"alpha{ExperimentConfig.node_emb_update_weight}")

    if ExperimentConfig.retrieval_type in ('vector', 'rrf'):
        parts.append(f"{ExperimentConfig.retrieval_type.upper()}")
        if ExperimentConfig.rerank_type == "ada":
            parts.append(ExperimentConfig.rerank_type)
        else:
            rc = ExperimentConfig.retrieval_config
            parts.append(f"{ExperimentConfig.rerank_type}-{rc['initial_candidates']}-{rc['topic_top_k']}-{rc['episode_top_k']}-{rc['fact_top_k']}")
    else:
        parts.append(ExperimentConfig.retrieval_type)
    parts.append(f"r{ExperimentConfig.hypergraph_retrieval_output_type}")

    if ExperimentConfig.use_reranker:
        parts.append("rerank")
    else:
        parts.append("wo-rerank")

    parts.append(f"{ExperimentConfig.answer_type}")

    return "_".join(parts)

ExperimentConfig.experiment_name = _build_experiment_name()
