"""
Test fixtures.

Key decisions:
- moto mocks AWS (S3, Bedrock) so tests never hit real AWS
- pytest-asyncio for async tests
- A real SQLite (in-memory via aiosqlite) replaces PostgreSQL — fast, no Docker needed
- fakeredis replaces Redis
"""
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from httpx import AsyncClient, ASGITransport

from app.main import app


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest_asyncio.fixture
async def client():
    """Async HTTP test client for the FastAPI app."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


@pytest.fixture
def mock_s3(monkeypatch):
    """Mock S3 upload/download so no AWS calls are made."""
    monkeypatch.setattr("app.services.s3.upload_bytes", lambda *a, **kw: None)
    monkeypatch.setattr("app.services.s3.download_bytes", lambda key: b"test pdf content")
    monkeypatch.setattr("app.services.s3.delete_object", lambda key: None)


@pytest.fixture
def mock_bedrock(monkeypatch):
    """Mock Bedrock embed to return a dummy 1024-dim vector."""
    dummy_vector = [0.01] * 1024

    async def fake_embed_texts(texts):
        return [dummy_vector for _ in texts]

    async def fake_embed_query(query):
        return dummy_vector

    monkeypatch.setattr("app.services.bedrock.embed_texts", fake_embed_texts)
    monkeypatch.setattr("app.services.bedrock.embed_query", fake_embed_query)
    monkeypatch.setattr("app.mcp.client.embed_query", fake_embed_query)


@pytest.fixture
def mock_opensearch(monkeypatch):
    """Mock OpenSearch so no real cluster is needed."""
    monkeypatch.setattr("app.services.opensearch.ensure_index", AsyncMock())
    monkeypatch.setattr("app.services.opensearch.bulk_index", AsyncMock(return_value=(5, [])))
    monkeypatch.setattr(
        "app.services.opensearch.hybrid_search",
        AsyncMock(return_value=[]),
    )


@pytest.fixture
def mock_redis(monkeypatch):
    """Mock Redis cache — always misses (returns None)."""
    monkeypatch.setattr("app.services.cache.cache_get", AsyncMock(return_value=None))
    monkeypatch.setattr("app.services.cache.cache_set", AsyncMock())
    monkeypatch.setattr("app.services.cache.cache_ping", AsyncMock(return_value=True))
    monkeypatch.setattr("app.api.routes.query.cache_get", AsyncMock(return_value=None))
    monkeypatch.setattr("app.api.routes.query.cache_set", AsyncMock())


@pytest.fixture
def mock_celery(monkeypatch):
    """Mock Celery task so no broker is needed in tests."""
    mock_task = MagicMock()
    mock_task.id = "test-task-id-123"
    monkeypatch.setattr(
        "app.api.routes.documents.ingest_document.delay",
        MagicMock(return_value=mock_task),
    )


@pytest.fixture
def mock_db(monkeypatch):
    """Mock database session — returns a simple AsyncMock."""
    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(return_value=MagicMock(
        scalar_one_or_none=MagicMock(return_value=None),
        scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[]))),
    ))
    mock_session.commit = AsyncMock()
    mock_session.add = MagicMock()

    async def override_get_db():
        yield mock_session

    from app.services.database import get_db
    app.dependency_overrides[get_db] = override_get_db
    yield mock_session
    app.dependency_overrides.clear()
