# Multi-Agent RAG Platform

> **Document intelligence API** — upload PDFs, ask natural-language questions,  
> get cited answers powered by a LangGraph multi-agent system backed by AWS Bedrock,  
> OpenSearch hybrid retrieval, and an MCP tool server.

---

## Architecture

```
                        ┌──────────────────────────────────────────┐
  Client                │         FastAPI  (port 8000)             │
    │                   │   /api/v1/documents  /api/v1/query       │
    │   HTTP            └───────────────┬──────────────────────────┘
    └──────────────────────────────────►│
                                        │ depends on
          ┌─────────────────────────────▼──────────────────────────┐
          │           LangGraph Agent                              │
          │   START → planner → tool_executor ←─┐                 │
          │                  └──► synthesiser   │ (loop)          │
          │                         └──► END    │                 │
          └──────────────┬──────────────────────┘                 │
                         │ HTTP tool calls                        │
          ┌──────────────▼───────────────────────────────────┐    │
          │     MCP Server  (port 8001)                      │    │
          │  /tools/retrieve  /tools/list_documents          │    │
          │  /tools/sql_query                                │    │
          └───┬──────────────┬──────────────────────────┐    │    │
              │              │                          │    │    │
   ┌──────────▼──┐  ┌────────▼──────┐       ┌──────────▼─┐  │    │
   │ OpenSearch  │  │  PostgreSQL   │       │   Redis    │  │    │
   │  kNN + BM25 │  │  metadata     │       │   cache    │  │    │
   └──────────┬──┘  └───────────────┘       └────────────┘  │    │
              │                                              │    │
   ┌──────────▼──────────────────────────────────────────┐  │    │
   │  AWS Bedrock (Titan Embed v2 + Claude 3.5 Sonnet)   │  │    │
   └─────────────────────────────────────────────────────┘  │    │
                                                             │    │
   ┌──────────────────────────────────────────────────────┐  │    │
   │  Celery Worker  ←── Redis broker                    │◄─┘    │
   │  S3 download → chunk → embed → index                │       │
   └──────────────────────────────────────────────────────┘       │
              │                                                    │
   ┌──────────▼──────────┐                                        │
   │  AWS S3             │                                        │
   │  (raw documents)    │                                        │
   └─────────────────────┘                                        │
```

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| API framework | FastAPI 0.115 | Async, typed, auto-docs |
| Agent orchestration | LangGraph 0.2 | Stateful multi-step agent loops |
| LLM + embeddings | AWS Bedrock (Claude 3.5, Titan Embed v2) | No OpenAI dependency, enterprise-ready |
| Tool protocol | MCP (Model Context Protocol) | Decoupled tool server, agent-agnostic |
| Vector search | OpenSearch 2.x (HNSW/FAISS) | Hybrid dense + sparse retrieval |
| Relational DB | PostgreSQL 16 + SQLAlchemy async | Metadata, chunk tracking, sessions |
| Task queue | Celery + Redis | Async document ingestion |
| Object storage | AWS S3 | Raw document persistence |
| Cache | Redis | Query result caching |
| Migrations | Alembic | Schema version control |
| PDF parsing | PyMuPDF | Fast, accurate text + page extraction |
| Tokenization | tiktoken | Token-accurate chunk sizing |
| Testing | pytest-asyncio + moto | Async tests, AWS mocking |
| CI/CD | GitHub Actions → ECS | Lint → test → build → deploy |

---

## Quick Start (Local)

```bash
# 1. Clone and set up environment
git clone https://github.com/your-username/multi-agent-rag
cd multi-agent-rag
cp .env.example .env   # fill in your AWS credentials

# 2. Start all services
docker compose up -d

# 3. Run database migrations
docker compose exec api alembic upgrade head

# 4. Run tests
pytest tests/ -v

# 5. Open API docs
open http://localhost:8000/docs
```

---

## API Reference

### Upload a document
```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -H "X-Tenant-Id: acme-corp" \
  -F "file=@report.pdf"
# → { "document_id": "...", "status": "pending", "task_id": "..." }
```

### Check indexing status
```bash
curl http://localhost:8000/api/v1/documents/{document_id} \
  -H "X-Tenant-Id: acme-corp"
# → { "status": "indexed", "chunk_count": 42 }
```

### Query the knowledge base
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key risks identified?", "tenant_id": "acme-corp"}'
# → { "answer": "...", "sources": [...], "latency_ms": 1240 }
```

### Streaming query (SSE)
```bash
curl -X POST http://localhost:8000/api/v1/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarise the financials", "tenant_id": "acme-corp", "stream": true}'
# → SSE stream of token/tool_call/done events
```

---

## Key Design Decisions

### Hybrid Retrieval (kNN + BM25)
Pure vector search misses exact keyword matches (product codes, names, IDs).
Pure BM25 misses semantic similarity. Weighted fusion of both gives best of both worlds.
Weights are configurable: `KNN_WEIGHT=0.7, BM25_WEIGHT=0.3`.

### MCP as Tool Protocol
The agent calls tools over HTTP rather than importing them directly.
This means tools can be updated/scaled independently, and the agent code
stays clean — it just knows about tool names and schemas.

### Async Everything
FastAPI + asyncpg + redis.asyncio + opensearch-py async = zero thread blocking.
The only sync code is the Celery worker (which runs in a subprocess).

### Tenant Isolation
Every S3 key is prefixed `tenants/{tenant_id}/`, every DB query filters by `tenant_id`,
and every OpenSearch query includes a `term` filter on `tenant_id`.
One mis-configured query cannot leak another tenant's data.

### LangGraph Loop Guard
`max_agent_iterations` (default 10) prevents infinite tool-call loops.
The `should_continue` edge forces synthesis after the limit is reached.

---

## Project Structure

```
multi-agent-rag/
├── app/
│   ├── main.py              # FastAPI entry point
│   ├── mcp_main.py          # MCP server entry point
│   ├── config.py            # Pydantic settings
│   ├── models/
│   │   ├── db.py            # SQLAlchemy ORM models
│   │   └── schemas.py       # Pydantic request/response schemas
│   ├── api/routes/
│   │   ├── documents.py     # Upload, status, delete
│   │   ├── query.py         # Sync + streaming query
│   │   └── health.py        # Health check
│   ├── agents/
│   │   └── graph.py         # LangGraph agent (planner → tools → synthesiser)
│   ├── mcp/
│   │   ├── server.py        # MCP tool server (FastAPI)
│   │   └── client.py        # MCP client (LangChain StructuredTools)
│   ├── services/
│   │   ├── bedrock.py       # AWS Bedrock embed + chat
│   │   ├── opensearch.py    # Index management + hybrid search
│   │   ├── s3.py            # Upload/download/delete
│   │   ├── cache.py         # Redis async cache
│   │   ├── chunker.py       # Text splitting + token counting
│   │   └── database.py      # Async SQLAlchemy session
│   └── workers/
│       └── ingest.py        # Celery document ingestion task
├── tests/
│   ├── conftest.py          # Shared fixtures
│   ├── test_unit.py         # Unit tests (no infra)
│   └── test_integration.py  # API integration tests
├── alembic/                 # DB migrations
├── docker/                  # Dockerfiles
├── .github/workflows/       # CI/CD pipeline
├── docker-compose.yml
├── requirements.txt
└── .env.example
```
