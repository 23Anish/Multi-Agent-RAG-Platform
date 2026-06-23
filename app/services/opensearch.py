"""
OpenSearch service.

Index mapping uses:
  - `embedding`  : knn_vector (dense retrieval via FAISS/HNSW)
  - `text`       : text with standard analyser (BM25 / sparse retrieval)

Hybrid search fuses both scores using a weighted sum before returning.
"""
import logging
import uuid
from dataclasses import dataclass

from opensearchpy import AsyncOpenSearch, helpers

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

INDEX_MAPPING = {
    "settings": {
        "index": {
            "knn": True,
            "knn.algo_param.ef_search": 512,
            "number_of_shards": 1,
            "number_of_replicas": 1,
        }
    },
    "mappings": {
        "properties": {
            "tenant_id":    {"type": "keyword"},
            "document_id":  {"type": "keyword"},
            "chunk_id":     {"type": "keyword"},
            "chunk_index":  {"type": "integer"},
            "filename":     {"type": "keyword"},
            "text":         {"type": "text", "analyzer": "standard"},
            "page_num":     {"type": "integer"},
            "token_count":  {"type": "integer"},
            "embedding": {
                "type": "knn_vector",
                "dimension": settings.bedrock_embed_dimensions,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "faiss",
                    "parameters": {"ef_construction": 256, "m": 48},
                },
            },
        }
    },
}


@dataclass
class SearchResult:
    chunk_id: str
    document_id: str
    filename: str
    text: str
    score: float
    page_num: int | None


def _get_client() -> AsyncOpenSearch:
    return AsyncOpenSearch(
        hosts=[{"host": settings.opensearch_host, "port": settings.opensearch_port}],
        http_auth=(settings.opensearch_user, settings.opensearch_password),
        use_ssl=settings.opensearch_use_ssl,
        verify_certs=False,
        http_compress=True,
    )


async def ensure_index() -> None:
    """Create the index if it doesn't exist (idempotent)."""
    client = _get_client()
    try:
        exists = await client.indices.exists(index=settings.opensearch_index)
        if not exists:
            await client.indices.create(index=settings.opensearch_index, body=INDEX_MAPPING)
            logger.info("Created OpenSearch index: %s", settings.opensearch_index)
    finally:
        await client.close()


async def bulk_index(documents: list[dict]) -> tuple[int, list]:
    """
    Bulk-index a list of chunk dicts.
    Each dict must contain: tenant_id, document_id, chunk_id, filename,
    text, embedding, chunk_index, page_num, token_count.
    """
    client = _get_client()
    try:
        actions = [
            {
                "_index": settings.opensearch_index,
                "_id": doc["chunk_id"],
                "_source": doc,
            }
            for doc in documents
        ]
        success, errors = await helpers.async_bulk(client, actions, raise_on_error=False)
        if errors:
            logger.error("Bulk index errors: %s", errors)
        return success, errors
    finally:
        await client.close()


async def hybrid_search(
    query_text: str,
    query_vector: list[float],
    tenant_id: str,
    top_k: int = 8,
) -> list[SearchResult]:
    """
    Hybrid retrieval: weighted fusion of kNN (dense) + BM25 (sparse).

    Strategy:
      1. Run kNN search → top_k * 2 results with cosine scores
      2. Run BM25 match  → top_k * 2 results with BM25 scores
      3. Normalise each score set to [0,1]
      4. Fuse: final_score = knn_weight * knn_score + bm25_weight * bm25_score
      5. Deduplicate by chunk_id, return top_k
    """
    client = _get_client()
    fetch_k = top_k * 2

    try:
        # ── kNN query ─────────────────────────────────────────────────────────
        knn_resp = await client.search(
            index=settings.opensearch_index,
            body={
                "size": fetch_k,
                "query": {
                    "bool": {
                        "filter": [{"term": {"tenant_id": tenant_id}}],
                        "must": [
                            {
                                "knn": {
                                    "embedding": {
                                        "vector": query_vector,
                                        "k": fetch_k,
                                    }
                                }
                            }
                        ],
                    }
                },
            },
        )

        # ── BM25 query ────────────────────────────────────────────────────────
        bm25_resp = await client.search(
            index=settings.opensearch_index,
            body={
                "size": fetch_k,
                "query": {
                    "bool": {
                        "filter": [{"term": {"tenant_id": tenant_id}}],
                        "must": [{"match": {"text": {"query": query_text, "operator": "or"}}}],
                    }
                },
            },
        )

    finally:
        await client.close()

    # ── Score fusion ──────────────────────────────────────────────────────────
    knn_hits = {h["_id"]: h for h in knn_resp["hits"]["hits"]}
    bm25_hits = {h["_id"]: h for h in bm25_resp["hits"]["hits"]}

    def _normalise(hits: dict) -> dict[str, float]:
        scores = {cid: h["_score"] for cid, h in hits.items()}
        mx = max(scores.values(), default=1.0)
        mn = min(scores.values(), default=0.0)
        rng = mx - mn or 1.0
        return {cid: (s - mn) / rng for cid, s in scores.items()}

    knn_norm = _normalise(knn_hits)
    bm25_norm = _normalise(bm25_hits)

    all_ids = set(knn_norm) | set(bm25_norm)
    fused: list[tuple[str, float]] = []
    for cid in all_ids:
        score = (
            settings.knn_weight * knn_norm.get(cid, 0.0)
            + settings.bm25_weight * bm25_norm.get(cid, 0.0)
        )
        fused.append((cid, score))

    fused.sort(key=lambda x: x[1], reverse=True)

    results: list[SearchResult] = []
    for cid, score in fused[:top_k]:
        src = (knn_hits.get(cid) or bm25_hits[cid])["_source"]
        results.append(
            SearchResult(
                chunk_id=cid,
                document_id=src["document_id"],
                filename=src["filename"],
                text=src["text"],
                score=round(score, 4),
                page_num=src.get("page_num"),
            )
        )

    return results


async def delete_by_document(document_id: str) -> int:
    """Delete all chunks belonging to a document (used on document deletion)."""
    client = _get_client()
    try:
        resp = await client.delete_by_query(
            index=settings.opensearch_index,
            body={"query": {"term": {"document_id": document_id}}},
        )
        return resp.get("deleted", 0)
    finally:
        await client.close()
