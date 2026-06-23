import json
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.graph import run_agent, stream_agent
from app.models.db import AgentSession
from app.models.schemas import QueryRequest, QueryResponse, SourceChunk
from app.services.cache import cache_get, cache_set
from app.services.database import get_db

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def query(
    req: QueryRequest,
    db: AsyncSession = Depends(get_db),
) -> QueryResponse:
    """
    Run the multi-agent RAG pipeline for a query.

    Cache strategy:
      - Cache key = hash(tenant_id + query)
      - TTL = 1 hour
      - Cached responses skip the agent entirely (fast path)
    """
    cache_key = f"query:{req.tenant_id}:{hash(req.query.lower().strip())}"
    cached = await cache_get(cache_key)
    if cached:
        return QueryResponse(**cached)

    session_id = uuid.uuid4()

    # ── Persist session (running) ──────────────────────────────────────────────
    agent_session = AgentSession(
        id=session_id,
        tenant_id=req.tenant_id,
        query=req.query,
        status="running",
    )
    db.add(agent_session)
    await db.commit()

    # ── Run agent ──────────────────────────────────────────────────────────────
    try:
        result = await run_agent(query=req.query, tenant_id=req.tenant_id, session_id=session_id)
    except Exception as exc:
        agent_session.status = "failed"
        await db.commit()
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc

    # ── Build response ─────────────────────────────────────────────────────────
    sources = [
        SourceChunk(
            chunk_id=s.get("chunk_id", ""),
            document_id=uuid.UUID(s["document_id"]) if "document_id" in s else uuid.uuid4(),
            filename=s.get("source_meta", "unknown").split(" (")[0],
            text="",   # text is in the agent answer; omit here to keep response lean
            score=0.0,
            page_num=None,
        )
        for s in result["sources"]
        if "chunk_id" in s
    ]

    response = QueryResponse(
        session_id=session_id,
        query=req.query,
        answer=result["final_answer"],
        sources=sources,
        tool_calls=result["tool_calls_log"],
        latency_ms=result["latency_ms"],
    )

    # ── Update session ─────────────────────────────────────────────────────────
    agent_session.final_answer = result["final_answer"]
    agent_session.tool_calls = result["tool_calls_log"]
    agent_session.status = "completed"
    agent_session.latency_ms = result["latency_ms"]
    await db.commit()

    # ── Cache successful result ────────────────────────────────────────────────
    await cache_set(cache_key, response.model_dump(mode="json"))

    return response


@router.post("/stream")
async def query_stream(req: QueryRequest) -> StreamingResponse:
    """
    SSE streaming endpoint. Yields JSON-encoded AgentEvent objects.

    Client reads:
      data: {"event": "token", "data": "Hello"}
      data: {"event": "tool_call", "data": {"tool": "retrieve_chunks", "args": {...}}}
      data: {"event": "done", "data": {"sources": [...]}}
      data: {"event": "error", "data": "..."}
    """

    async def event_generator():
        try:
            async for event in stream_agent(req.query, req.tenant_id):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'event': 'error', 'data': str(exc)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )
