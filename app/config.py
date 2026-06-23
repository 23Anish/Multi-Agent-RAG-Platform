from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── App ──────────────────────────────────────────────────────────────────
    app_name: str = "multi-agent-rag"
    environment: Literal["local", "staging", "production"] = "local"
    log_level: str = "INFO"
    secret_key: str = Field(..., min_length=32)

    # ── AWS ──────────────────────────────────────────────────────────────────
    aws_region: str = "us-east-1"
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    s3_bucket: str = "rag-documents"
    bedrock_embed_model: str = "amazon.titan-embed-text-v2:0"
    bedrock_chat_model: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    bedrock_embed_dimensions: int = 1024

    # ── OpenSearch ───────────────────────────────────────────────────────────
    opensearch_host: str = "localhost"
    opensearch_port: int = 9200
    opensearch_user: str = "admin"
    opensearch_password: str = "admin"
    opensearch_index: str = "rag_documents"
    opensearch_use_ssl: bool = False

    # ── PostgreSQL ───────────────────────────────────────────────────────────
    postgres_dsn: str = "postgresql+asyncpg://rag:rag@localhost:5432/ragdb"
    postgres_pool_size: int = 10
    postgres_max_overflow: int = 5

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 3600

    # ── Celery ───────────────────────────────────────────────────────────────
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # ── Agent ────────────────────────────────────────────────────────────────
    max_agent_iterations: int = 10
    retriever_top_k: int = 8
    rerank_top_k: int = 4
    bm25_weight: float = 0.3
    knn_weight: float = 0.7

    # ── MCP ──────────────────────────────────────────────────────────────────
    mcp_server_host: str = "localhost"
    mcp_server_port: int = 8001

    @model_validator(mode="after")
    def validate_weights(self) -> "Settings":
        total = round(self.bm25_weight + self.knn_weight, 2)
        if total != 1.0:
            raise ValueError(f"bm25_weight + knn_weight must equal 1.0, got {total}")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
