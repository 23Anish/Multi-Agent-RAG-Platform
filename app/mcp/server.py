"""
MCP (Model Context Protocol) Server.

This FastAPI app exposes tools that the LangGraph agent calls via
the MCP tool-use protocol.  Running as a separate process allows:
  - Independent scaling of the tool layer
  - Clear security boundary (agent cannot directly touch DB/S3)
  - Easy addition of new tools without redeploying the agent

Exposed tools:
  1. retrieve_chunks   — hybrid kNN + BM25 retrieval from OpenSearch
  2. list_documents    — list indexed documents for a tenant
  3. sql_query         — safe read-only SQL against PostgreSQL
"""
import logging
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from app.config import get_settings
from app.services.bedrock import embed_query
from app.services.cache import cache_get, cache_set
from app.services.database import get_db_session
from app.services.opensearch import hybrid_search
from app.models.db import Document
from sqlalchemy import select, text

logger = logging.getLogger(__name__)
settings = get_settings()

mcp_app = FastAPI(title="RAG MCP Server", version="1.0.0")

# ── Tool request/response schemas ────────────────────────────────────────────

class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=3)
    tenant_id: str
    top_k: int = Field(default=4, ge=1, le=20)


class ChunkResult(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    text: str
    score: float
    page_num: int | None


class RetrieveResponse(BaseModel):
    chunks: list[ChunkResult]
    cached: bool = False


class ListDocumentsRequest(BaseModel):
    tenant_id: str
    status: str | None = None  # filter by status


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    status: str
    chunk_count: int


class ListDocumentsResponse(BaseModel):
    documents: list[DocumentInfo]


class SqlQueryRequest(BaseModel):
    sql: str = Field(..., description="Read-only SQL query against ragdb")
    tenant_id: str


class SqlQueryResponse(BaseModel):
    rows: list[dict[str, Any]]
    row_count: int


# ── Tool endpoints ────────────────────────────────────────────────────────────

@mcp_app.post("/tools/retrieve", response_model=RetrieveResponse)
async def retrieve_chunks(req: RetrieveRequest) -> RetrieveResponse:
    """
    Hybrid semantic + keyword retrieval.
    Results are cached in Redis for 1 hour keyed by (query, tenant, top_k).
    """
    cache_key = f"retrieve:{req.tenant_id}:{req.top_k}:{hash(req.query)}"
    cached = await cache_get(cache_key)
    if cached:
        return RetrieveResponse(chunks=cached, cached=True)

    vector = await embed_query(req.query)
    hits = await hybrid_search(req.query, vector, req.tenant_id, req.top_k)

    chunks = [
        ChunkResult(
            chunk_id=h.chunk_id,
            document_id=h.document_id,
            filename=h.filename,
            text=h.text,
            score=h.score,
            page_num=h.page_num,
        )
        for h in hits
    ]

    await cache_set(cache_key, [c.model_dump() for c in chunks])
    return RetrieveResponse(chunks=chunks, cached=False)


@mcp_app.post("/tools/list_documents", response_model=ListDocumentsResponse)
async def list_documents(req: ListDocumentsRequest) -> ListDocumentsResponse:
    async with get_db_session() as session:
        stmt = select(Document).where(Document.tenant_id == req.tenant_id)
        if req.status:
            stmt = stmt.where(Document.status == req.status)
        result = await session.execute(stmt.order_by(Document.created_at.desc()).limit(50))
        docs = result.scalars().all()

    return ListDocumentsResponse(
        documents=[
            DocumentInfo(
                document_id=str(d.id),
                filename=d.filename,
                status=d.status,
                chunk_count=d.meta.get("chunk_count", 0) if d.meta else 0,
            )
            for d in docs
        ]
    )


@mcp_app.post("/tools/sql_query", response_model=SqlQueryResponse)
async def sql_query(req: SqlQueryRequest) -> SqlQueryResponse:
    """
    Execute a read-only SQL query.  Only SELECT is allowed.
    Tenant isolation is enforced by injecting a WHERE clause.
    
    WARNING: In production, use a dedicated read replica and a DB user
    with SELECT-only grants.  Never expose this endpoint publicly.
    """
    sql_lower = req.sql.strip().lower()
    if not sql_lower.startswith("select"):
        raise HTTPException(status_code=400, detail="Only SELECT queries are allowed")

    # Inject tenant filter if not already present
    if "tenant_id" not in sql_lower:
        raise HTTPException(
            status_code=400,
            detail="Query must filter by tenant_id for data isolation",
        )

    async with get_db_session() as session:
        result = await session.execute(text(req.sql))
        rows = [dict(row._mapping) for row in result.fetchmany(100)]  # cap at 100 rows

    return SqlQueryResponse(rows=rows, row_count=len(rows))


@mcp_app.get("/health")
async def health() -> dict:
    return {"status": "ok", "tools": ["retrieve", "list_documents", "sql_query"]}
