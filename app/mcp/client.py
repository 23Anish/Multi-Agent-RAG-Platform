"""
MCP Client — wraps each MCP server endpoint as a LangChain StructuredTool.

The agent imports these tools and passes them to LangGraph.
Using HTTP (not in-process calls) means the agent and MCP server
can scale independently and be in separate containers.
"""
import logging
from typing import Any

import httpx
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

MCP_BASE = f"http://{settings.mcp_server_host}:{settings.mcp_server_port}"
TIMEOUT = httpx.Timeout(30.0)


async def _post(endpoint: str, payload: dict) -> dict:
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(f"{MCP_BASE}{endpoint}", json=payload)
        resp.raise_for_status()
        return resp.json()


# ── Tool input schemas ────────────────────────────────────────────────────────

class RetrieveInput(BaseModel):
    query: str = Field(..., description="Natural language query to retrieve relevant document chunks")
    tenant_id: str = Field(..., description="Tenant/organization identifier for data isolation")
    top_k: int = Field(default=4, description="Number of chunks to return (1-20)")


class ListDocumentsInput(BaseModel):
    tenant_id: str = Field(..., description="Tenant/organization identifier")
    status: str | None = Field(default=None, description="Filter by status: pending/processing/indexed/failed")


class SqlQueryInput(BaseModel):
    sql: str = Field(..., description="Read-only SELECT SQL query. Must include tenant_id filter.")
    tenant_id: str = Field(..., description="Tenant/organization identifier for data isolation")


# ── Tool implementations ──────────────────────────────────────────────────────

async def retrieve_chunks_tool(query: str, tenant_id: str, top_k: int = 4) -> str:
    """Retrieve semantically relevant document chunks using hybrid search."""
    try:
        result = await _post("/tools/retrieve", {"query": query, "tenant_id": tenant_id, "top_k": top_k})
        chunks = result.get("chunks", [])
        if not chunks:
            return "No relevant chunks found for the query."

        parts = []
        for i, c in enumerate(chunks, 1):
            parts.append(
                f"[{i}] Source: {c['filename']} (page {c.get('page_num', '?')}, score={c['score']:.3f})\n"
                f"    chunk_id={c['chunk_id']}\n"
                f"    {c['text'][:600]}{'...' if len(c['text']) > 600 else ''}"
            )
        return "\n\n".join(parts)
    except Exception as exc:
        logger.error("retrieve_chunks_tool failed: %s", exc)
        return f"ERROR: retrieve failed — {exc}"


async def list_documents_tool(tenant_id: str, status: str | None = None) -> str:
    """List documents available for retrieval."""
    try:
        result = await _post("/tools/list_documents", {"tenant_id": tenant_id, "status": status})
        docs = result.get("documents", [])
        if not docs:
            return "No documents found."
        lines = [f"- {d['filename']} | status={d['status']} | chunks={d['chunk_count']} | id={d['document_id']}" for d in docs]
        return f"Found {len(docs)} document(s):\n" + "\n".join(lines)
    except Exception as exc:
        return f"ERROR: list_documents failed — {exc}"


async def sql_query_tool(sql: str, tenant_id: str) -> str:
    """Execute a read-only SQL query for structured data."""
    try:
        result = await _post("/tools/sql_query", {"sql": sql, "tenant_id": tenant_id})
        rows = result.get("rows", [])
        if not rows:
            return "Query returned no rows."
        header = " | ".join(rows[0].keys())
        lines = [header, "-" * len(header)]
        for row in rows[:20]:
            lines.append(" | ".join(str(v) for v in row.values()))
        if result["row_count"] > 20:
            lines.append(f"... ({result['row_count']} total rows, showing 20)")
        return "\n".join(lines)
    except Exception as exc:
        return f"ERROR: sql_query failed — {exc}"


# ── Assemble LangChain StructuredTools ────────────────────────────────────────

def build_mcp_tools(tenant_id: str) -> list[StructuredTool]:
    """
    Returns a list of tools pre-bound to `tenant_id`.
    The agent doesn't need to know about tenant isolation — it's baked in.
    """
    async def _retrieve(query: str, top_k: int = 4) -> str:
        return await retrieve_chunks_tool(query, tenant_id, top_k)

    async def _list_docs(status: str | None = None) -> str:
        return await list_documents_tool(tenant_id, status)

    async def _sql(sql: str) -> str:
        return await sql_query_tool(sql, tenant_id)

    class RetrieveOnlyInput(BaseModel):
        query: str = Field(..., description="Natural language search query")
        top_k: int = Field(default=4)

    class ListOnlyInput(BaseModel):
        status: str | None = Field(default=None)

    class SqlOnlyInput(BaseModel):
        sql: str = Field(..., description="Read-only SELECT SQL with tenant_id filter")

    return [
        StructuredTool.from_function(
            coroutine=_retrieve,
            name="retrieve_chunks",
            description=(
                "Search the knowledge base using hybrid semantic + keyword retrieval. "
                "Use this when answering questions that require document context. "
                "Returns ranked text chunks with source citations."
            ),
            args_schema=RetrieveOnlyInput,
        ),
        StructuredTool.from_function(
            coroutine=_list_docs,
            name="list_documents",
            description=(
                "List all indexed documents in the knowledge base. "
                "Use this to understand what content is available before retrieval."
            ),
            args_schema=ListOnlyInput,
        ),
        StructuredTool.from_function(
            coroutine=_sql,
            name="sql_query",
            description=(
                "Run a read-only SQL query for structured metadata about documents and chunks. "
                "Example: SELECT filename, status FROM documents WHERE tenant_id='x'. "
                "Always include tenant_id in the WHERE clause."
            ),
            args_schema=SqlOnlyInput,
        ),
    ]
