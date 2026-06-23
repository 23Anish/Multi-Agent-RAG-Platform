import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.db import Chunk, Document
from app.models.schemas import DocumentStatus, DocumentUploadResponse
from app.services.database import get_db
from app.services.s3 import delete_object, s3_key, upload_bytes
from app.workers.ingest import ingest_document

router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "text/plain",
    "text/markdown",
    "application/octet-stream",
}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


def _require_tenant(x_tenant_id: Annotated[str | None, Header()] = None) -> str:
    if not x_tenant_id:
        raise HTTPException(status_code=400, detail="X-Tenant-Id header is required")
    return x_tenant_id


@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    file: UploadFile = File(...),
    tenant_id: str = Depends(_require_tenant),
    db: AsyncSession = Depends(get_db),
) -> DocumentUploadResponse:
    """
    Upload a document for indexing.

    Flow:
      1. Validate file type and size
      2. Upload raw bytes to S3  (tenant-scoped prefix)
      3. Persist Document record in PostgreSQL (status=pending)
      4. Enqueue Celery ingest task → returns 202 Accepted immediately
    """
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content type: {file.content_type}. Allowed: {ALLOWED_CONTENT_TYPES}",
        )

    data = await file.read()
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large (max {MAX_FILE_SIZE // 1024 // 1024} MB)")

    doc_id = uuid.uuid4()
    key = s3_key(tenant_id, doc_id, file.filename)

    # Upload to S3
    upload_bytes(key, data, file.content_type)

    # Persist metadata
    doc = Document(
        id=doc_id,
        tenant_id=tenant_id,
        filename=file.filename,
        s3_key=key,
        content_type=file.content_type,
        size_bytes=len(data),
        status="pending",
    )
    db.add(doc)
    await db.commit()

    # Fire-and-forget Celery task
    task = ingest_document.delay(
        document_id=str(doc_id),
        tenant_id=tenant_id,
        s3_key=key,
        content_type=file.content_type,
    )

    return DocumentUploadResponse(
        document_id=doc_id,
        filename=file.filename,
        status="pending",
        task_id=task.id,
    )


@router.get("/{document_id}", response_model=DocumentStatus)
async def get_document_status(
    document_id: uuid.UUID,
    tenant_id: str = Depends(_require_tenant),
    db: AsyncSession = Depends(get_db),
) -> DocumentStatus:
    result = await db.execute(
        select(Document).where(Document.id == document_id, Document.tenant_id == tenant_id)
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    chunk_count_result = await db.execute(
        select(Chunk).where(Chunk.document_id == document_id)
    )
    chunks = chunk_count_result.scalars().all()

    return DocumentStatus(
        document_id=doc.id,
        filename=doc.filename,
        status=doc.status,
        chunk_count=len(chunks),
        created_at=doc.created_at,
        updated_at=doc.updated_at,
    )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: uuid.UUID,
    tenant_id: str = Depends(_require_tenant),
    db: AsyncSession = Depends(get_db),
) -> None:
    result = await db.execute(
        select(Document).where(Document.id == document_id, Document.tenant_id == tenant_id)
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Delete from S3
    try:
        delete_object(doc.s3_key)
    except Exception:
        pass  # best-effort; DB delete proceeds regardless

    # Cascade deletes chunks via FK
    await db.delete(doc)
    await db.commit()
