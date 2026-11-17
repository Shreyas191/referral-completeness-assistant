import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


# Cheapest current tier per provider; set LLM_MODEL to override (e.g. claude-sonnet-5-5, gpt-6.1-sol).
DEFAULT_MODELS = {"anthropic": "claude-haiku-5-5", "openai": "gpt-6-luna"}


@dataclass(frozen=True)
class Settings:
    llm_provider: str = os.getenv("LLM_PROVIDER", "anthropic").lower()
    llm_model: str = os.getenv("LLM_MODEL") or DEFAULT_MODELS.get(os.getenv("LLM_PROVIDER", "anthropic").lower(), "")
    chroma_dir: str = os.getenv("CHROMA_DIR", "data/chroma")
    top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    absence_top_k: int = 20          # wider search before declaring an item Missing
    quote_match_threshold: int = 90  # rapidfuzz partial_ratio needed for a citation to count


settings = Settings()
