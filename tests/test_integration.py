"""
Integration tests for the FastAPI endpoints.
All external dependencies (DB, S3, Celery, Redis, OpenSearch) are mocked.
"""
import uuid
import pytest
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime, timezone


pytestmark = pytest.mark.asyncio


class TestHealth:
    async def test_health_ok(self, client, mock_redis):
        """Health endpoint should return 200 even with mocked infra."""
        from app.services.database import get_db_session
        from unittest.mock import patch, AsyncMock, MagicMock
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def mock_db_session():
            mock = AsyncMock()
            mock.execute = AsyncMock(return_value=MagicMock())
            yield mock

        with patch("app.api.routes.health.get_db_session", mock_db_session):
            resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("ok", "degraded")
        assert "postgres" in data["checks"]
        assert "redis" in data["checks"]


class TestDocumentUpload:
    async def test_upload_missing_tenant_header(self, client):
        resp = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.txt", b"hello world", "text/plain")},
        )
        assert resp.status_code == 400

    async def test_upload_unsupported_content_type(self, client, mock_db):
        resp = await client.post(
            "/api/v1/documents/upload",
            headers={"X-Tenant-Id": "tenant1"},
            files={"file": ("test.exe", b"MZ\x90", "application/x-msdownload")},
        )
        assert resp.status_code == 415

    async def test_upload_success(self, client, mock_s3, mock_celery, mock_db):
        resp = await client.post(
            "/api/v1/documents/upload",
            headers={"X-Tenant-Id": "tenant1"},
            files={"file": ("notes.txt", b"This is a test document with enough content.", "text/plain")},
        )
        assert resp.status_code == 202
        data = resp.json()
        assert data["status"] == "pending"
        assert data["task_id"] == "test-task-id-123"
        assert "document_id" in data

    async def test_upload_file_too_large(self, client, mock_db):
        big_data = b"x" * (51 * 1024 * 1024)  # 51 MB
        resp = await client.post(
            "/api/v1/documents/upload",
            headers={"X-Tenant-Id": "tenant1"},
            files={"file": ("big.txt", big_data, "text/plain")},
        )
        assert resp.status_code == 413


class TestDocumentStatus:
    async def test_get_status_not_found(self, client, mock_db):
        doc_id = uuid.uuid4()
        resp = await client.get(
            f"/api/v1/documents/{doc_id}",
            headers={"X-Tenant-Id": "tenant1"},
        )
        assert resp.status_code == 404

    async def test_get_status_found(self, client):
        from app.services.database import get_db
        from app.models.db import Document

        doc_id = uuid.uuid4()
        fake_doc = MagicMock(spec=Document)
        fake_doc.id = doc_id
        fake_doc.filename = "test.pdf"
        fake_doc.status = "indexed"
        fake_doc.tenant_id = "tenant1"
        fake_doc.meta = {"chunk_count": 10}
        fake_doc.created_at = datetime.now(tz=timezone.utc)
        fake_doc.updated_at = datetime.now(tz=timezone.utc)

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=MagicMock(
            scalar_one_or_none=MagicMock(return_value=fake_doc),
            scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[]))),
        ))

        async def override_get_db():
            yield mock_session

        from app.main import app
        app.dependency_overrides[get_db] = override_get_db

        from httpx import AsyncClient, ASGITransport
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get(
                f"/api/v1/documents/{doc_id}",
                headers={"X-Tenant-Id": "tenant1"},
            )

        app.dependency_overrides.clear()
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "indexed"


class TestMcpServer:
    async def test_mcp_health(self):
        from httpx import AsyncClient, ASGITransport
        from app.mcp.server import mcp_app

        async with AsyncClient(transport=ASGITransport(app=mcp_app), base_url="http://test") as ac:
            resp = await ac.get("/health")
        assert resp.status_code == 200
        assert "tools" in resp.json()

    async def test_retrieve_disallowed_without_tenant(self):
        from httpx import AsyncClient, ASGITransport
        from app.mcp.server import mcp_app

        async with AsyncClient(transport=ASGITransport(app=mcp_app), base_url="http://test") as ac:
            resp = await ac.post("/tools/retrieve", json={"query": "test"})
        # Missing tenant_id — should fail validation
        assert resp.status_code == 422

    async def test_sql_query_rejects_non_select(self):
        from httpx import AsyncClient, ASGITransport
        from app.mcp.server import mcp_app

        async with AsyncClient(transport=ASGITransport(app=mcp_app), base_url="http://test") as ac:
            resp = await ac.post(
                "/tools/sql_query",
                json={"sql": "DROP TABLE documents", "tenant_id": "t1"},
            )
        assert resp.status_code == 400
        assert "SELECT" in resp.json()["detail"]
