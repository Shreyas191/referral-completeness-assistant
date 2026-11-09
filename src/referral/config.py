import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    llm_model: str = os.getenv("LLM_MODEL", "claude-sonnet-5-5")
    chroma_dir: str = os.getenv("CHROMA_DIR", "data/chroma")
    top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    absence_top_k: int = 20          # wider search before declaring an item Missing
    quote_match_threshold: int = 90  # rapidfuzz partial_ratio needed for a citation to count


settings = Settings()
