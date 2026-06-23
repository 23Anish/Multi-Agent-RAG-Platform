"""
Celery worker: document ingestion pipeline.

Flow:
  1. Download raw file bytes from S3
  2. Extract text (PDF / plain text)
  3. Chunk text into token-bounded segments
  4. Embed each chunk via Bedrock Titan
  5. Bulk-index into OpenSearch
  6. Persist chunk metadata to PostgreSQL
  7. Update document status to `indexed` (or `failed`)

Why Celery?
  Ingestion is CPU + I/O heavy and can take 10-60 s per document.
  Doing it synchronously in FastAPI would block the event loop and
  give users a bad experience.  Celery decouples ingest from the
  HTTP lifecycle.
"""
import asyncio
import logging
import uuid
from typing import Any

from celery import Celery
from celery.utils.log import get_task_logger
from sqlalchemy import select, update

from app.config import get_settings
from app.models.db import Chunk, Document
from app.services.bedrock import embed_texts
from app.services.chunker import chunk_text, extract_text_from_pdf_bytes
from app.services.database import AsyncSessionLocal
from app.services.opensearch import bulk_index, ensure_index
from app.services.s3 import download_bytes

settings = get_settings()
logger = get_task_logger(__name__)

celery_app = Celery(
    "rag_worker",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    worker_prefetch_multiplier=1,  # one task at a time per worker process
    task_acks_late=True,           # only ack after completion → no data loss on crash
)


def _run_async(coro) -> Any:
    """Run an async coroutine from a synchronous Celery task."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@celery_app.task(
    bind=True,
    name="workers.ingest_document",
    max_retries=3,
    default_retry_delay=30,
    autoretry_for=(Exception,),
    retry_backoff=True,
)
def ingest_document(self, document_id: str, tenant_id: str, s3_key: str, content_type: str) -> dict:
    """
    Celery task: ingest one document.
    Called by the FastAPI upload endpoint after writing metadata to PG.
    """
    doc_uuid = uuid.UUID(document_id)
    logger.info("Starting ingestion for document_id=%s", document_id)

    try:
        result = _run_async(_ingest_async(doc_uuid, tenant_id, s3_key, content_type))
        return result
    except Exception as exc:
        logger.exception("Ingestion failed for document_id=%s: %s", document_id, exc)
        # Update document status to failed
        _run_async(_set_status(doc_uuid, "failed", error=str(exc)))
        raise self.retry(exc=exc)


async def _ingest_async(
    doc_uuid: uuid.UUID,
    tenant_id: str,
    s3_key: str,
    content_type: str,
) -> dict:
    # ── 1. Update status → processing ────────────────────────────────────────
    await _set_status(doc_uuid, "processing")

    # ── 2. Download from S3 ───────────────────────────────────────────────────
    data = download_bytes(s3_key)
    filename = s3_key.split("/")[-1]
    logger.info("Downloaded %d bytes from S3 key=%s", len(data), s3_key)

    # ── 3. Extract text pages ─────────────────────────────────────────────────
    if "pdf" in content_type:
        pages = extract_text_from_pdf_bytes(data)
    else:
        # Plain text, markdown, etc.
        pages = [(1, data.decode("utf-8", errors="replace"))]

    # ── 4. Chunk each page ────────────────────────────────────────────────────
    all_chunks: list[dict] = []
    for page_num, text in pages:
        chunks = chunk_text(text, doc_uuid, tenant_id, filename, page_num=page_num)
        all_chunks.extend(chunks)

    if not all_chunks:
        raise ValueError("No chunks produced — document may be empty or unreadable")

    logger.info("Produced %d chunks for document_id=%s", len(all_chunks), doc_uuid)

    # ── 5. Embed in micro-batches of 25 ──────────────────────────────────────
    BATCH = 25
    texts = [c["text"] for c in all_chunks]
    vectors: list[list[float]] = []
    for i in range(0, len(texts), BATCH):
        batch_vecs = await embed_texts(texts[i : i + BATCH])
        vectors.extend(batch_vecs)

    for chunk, vec in zip(all_chunks, vectors):
        chunk["embedding"] = vec

    # ── 6. Ensure OpenSearch index + bulk index ───────────────────────────────
    await ensure_index()
    success, errors = await bulk_index(all_chunks)
    logger.info("OpenSearch bulk index: %d ok, %d errors", success, len(errors))

    # ── 7. Persist chunk metadata to PostgreSQL ───────────────────────────────
    async with AsyncSessionLocal() as session:
        db_chunks = [
            Chunk(
                id=uuid.UUID(c["chunk_id"]),
                document_id=doc_uuid,
                tenant_id=tenant_id,
                opensearch_id=c["chunk_id"],
                chunk_index=c["chunk_index"],
                text=c["text"],
                token_count=c["token_count"],
                meta={"page_num": c.get("page_num")},
            )
            for c in all_chunks
        ]
        session.add_all(db_chunks)
        await session.execute(
            update(Document)
            .where(Document.id == doc_uuid)
            .values(status="indexed", meta={"chunk_count": len(db_chunks)})
        )
        await session.commit()

    logger.info("Ingestion complete for document_id=%s (%d chunks)", doc_uuid, len(db_chunks))
    return {"document_id": str(doc_uuid), "chunks": len(db_chunks)}


async def _set_status(doc_uuid: uuid.UUID, status: str, error: str | None = None) -> None:
    async with AsyncSessionLocal() as session:
        values: dict = {"status": status}
        if error:
            values["meta"] = {"error": error}
        await session.execute(update(Document).where(Document.id == doc_uuid).values(**values))
        await session.commit()
