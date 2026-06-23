import logging
import uuid
from functools import lru_cache

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@lru_cache(maxsize=1)
def _s3_client():
    kwargs: dict = {"region_name": settings.aws_region, "config": Config(signature_version="s3v4")}
    if settings.aws_access_key_id:
        kwargs["aws_access_key_id"] = settings.aws_access_key_id
        kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
    return boto3.client("s3", **kwargs)


def s3_key(tenant_id: str, document_id: uuid.UUID, filename: str) -> str:
    """
    Canonical S3 key layout:
      tenants/<tenant_id>/documents/<doc_id>/<filename>
    This enforces tenant isolation at the object prefix level.
    """
    return f"tenants/{tenant_id}/documents/{document_id}/{filename}"


def generate_presigned_upload(key: str, content_type: str, expires: int = 900) -> str:
    """Return a presigned PUT URL valid for `expires` seconds."""
    return _s3_client().generate_presigned_url(
        "put_object",
        Params={"Bucket": settings.s3_bucket, "Key": key, "ContentType": content_type},
        ExpiresIn=expires,
    )


def generate_presigned_download(key: str, expires: int = 3600) -> str:
    return _s3_client().generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.s3_bucket, "Key": key},
        ExpiresIn=expires,
    )


def upload_bytes(key: str, data: bytes, content_type: str) -> None:
    """Direct upload — used by the Celery worker after chunking."""
    _s3_client().put_object(
        Bucket=settings.s3_bucket,
        Key=key,
        Body=data,
        ContentType=content_type,
        ServerSideEncryption="AES256",
    )


def download_bytes(key: str) -> bytes:
    try:
        resp = _s3_client().get_object(Bucket=settings.s3_bucket, Key=key)
        return resp["Body"].read()
    except ClientError as exc:
        error_code = exc.response["Error"]["Code"]
        if error_code == "NoSuchKey":
            raise FileNotFoundError(f"S3 key not found: {key}") from exc
        raise


def delete_object(key: str) -> None:
    _s3_client().delete_object(Bucket=settings.s3_bucket, Key=key)
