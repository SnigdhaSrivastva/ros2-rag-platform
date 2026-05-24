from __future__ import annotations

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', case_sensitive=False)

    app_name: str = 'ROS2 RAG API'
    app_env: str = Field(default='development')
    log_level: str = Field(default='INFO')

    hf_model_name: str = Field(default='ArmaanDhande/rag_model_t5_AI')
    hf_token: str | None = Field(default=None)

    embedding_model_name: str = Field(default='sentence-transformers/all-MiniLM-L6-v2')

    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection: str = Field(default='ros2_rag_chunks')

    mongo_uri: str
    mongo_database: str = Field(default='ros2_rag')
    mongo_collection: str = Field(default='raw_data')

    retrieve_top_k: int = Field(default=6)
    max_context_chars: int = Field(default=6000)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
