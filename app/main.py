"""
Multi-Agent RAG Platform — FastAPI entry point.

Architecture:
  ┌─────────────────────────────────────────────────────┐
  │  FastAPI (port 8000)                                │
  │    /documents  ← upload, status, delete             │
  │    /query      ← sync + SSE streaming               │
  │    /health     ← dependency probes                  │
  └───────────────────────┬─────────────────────────────┘
                          │ HTTP
         ┌────────────────▼──────────────────┐
         │  MCP Server (port 8001)           │
         │    /tools/retrieve                │
         │    /tools/list_documents          │
         │    /tools/sql_query               │
         └───────────────────────────────────┘
"""
import logging
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import documents, health, query
from app.config import get_settings
from app.services.opensearch import ensure_index

settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks."""
    logger.info("Starting up — environment=%s", settings.environment)
    try:
        await ensure_index()
        logger.info("OpenSearch index ready")
    except Exception as exc:
        logger.warning("OpenSearch not reachable on startup (will retry): %s", exc)
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Multi-Agent RAG Platform",
    description="LangGraph + MCP + Bedrock + OpenSearch document intelligence API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Middleware ────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.environment == "local" else [],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_timing(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = int((time.perf_counter() - t0) * 1000)
    response.headers["X-Response-Time-Ms"] = str(ms)
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s: %s", request.url.path, exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(health.router)
app.include_router(documents.router, prefix="/api/v1")
app.include_router(query.router, prefix="/api/v1")


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
