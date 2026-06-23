import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, HttpUrl


# ── Documents ────────────────────────────────────────────────────────────────

class DocumentUploadResponse(BaseModel):
    document_id: uuid.UUID
    filename: str
    status: str
    task_id: str  # Celery task id


class DocumentStatus(BaseModel):
    document_id: uuid.UUID
    filename: str
    status: str
    chunk_count: int = 0
    created_at: datetime
    updated_at: datetime


# ── Query ────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2048)
    tenant_id: str = Field(..., min_length=1, max_length=64)
    top_k: int = Field(default=4, ge=1, le=20)
    stream: bool = False


class SourceChunk(BaseModel):
    chunk_id: str
    document_id: uuid.UUID
    filename: str
    text: str
    score: float
    page_num: int | None = None


class QueryResponse(BaseModel):
    session_id: uuid.UUID
    query: str
    answer: str
    sources: list[SourceChunk]
    tool_calls: list[dict[str, Any]]
    latency_ms: int


# ── Agent SSE streaming ───────────────────────────────────────────────────────

class AgentEvent(BaseModel):
    event: str  # "token" | "tool_call" | "source" | "done" | "error"
    data: Any


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    checks: dict[str, str]
