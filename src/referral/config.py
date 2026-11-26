import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


# Cheapest current tier per provider; set LLM_MODEL to override (e.g. claude-sonnet-5-5, gpt-6.1-sol).
DEFAULT_MODELS = {"anthropic": "claude-haiku-5-5", "openai": "gpt-6-luna"}


def _float_env(name: str) -> float | None:
    value = os.getenv(name, "").strip()
    return float(value) if value else None


@dataclass(frozen=True)
class Settings:
    llm_provider: str = os.getenv("LLM_PROVIDER", "anthropic").lower()
    llm_model: str = os.getenv("LLM_MODEL") or DEFAULT_MODELS.get(os.getenv("LLM_PROVIDER", "anthropic").lower(), "")
    chroma_dir: str = os.getenv("CHROMA_DIR", "data/chroma")
    top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    # Per-request limit. Without it the client waited ~50 min on one stalled response and blocked a dev run
    # (pw-dev-r2, case 39 of 40). A timed-out request is retried llm_max_retries times, then the run fails.
    llm_timeout_s: float = float(os.getenv("LLM_TIMEOUT_S", "120"))
    llm_max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "2"))
    absence_top_k: int = 20          # wider search before declaring an item Missing
    quote_match_threshold: int = 90  # rapidfuzz partial_ratio needed for a citation to count
    # USD per million tokens for LLM_MODEL, copied from the provider's pricing page. Unset = cost not reported.
    price_input_per_mtok: float | None = _float_env("LLM_PRICE_INPUT_PER_MTOK")
    price_output_per_mtok: float | None = _float_env("LLM_PRICE_OUTPUT_PER_MTOK")


settings = Settings()
