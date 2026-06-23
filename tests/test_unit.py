"""
Unit tests for core services.
These do NOT require any running infrastructure.
"""
import uuid
import pytest
from unittest.mock import AsyncMock, patch, MagicMock


# ── Chunker tests ──────────────────────────────────────────────────────────────

class TestChunker:
    def test_chunk_normal_text(self):
        from app.services.chunker import chunk_text
        text = "The quick brown fox jumps over the lazy dog. " * 200
        doc_id = uuid.uuid4()
        chunks = chunk_text(text, doc_id, "tenant1", "test.txt", page_num=1)
        assert len(chunks) >= 1
        for c in chunks:
            assert c["tenant_id"] == "tenant1"
            assert c["document_id"] == str(doc_id)
            assert c["token_count"] >= 10
            assert c["page_num"] == 1
            assert "chunk_id" in c
            assert len(c["text"]) > 0

    def test_chunk_empty_text(self):
        from app.services.chunker import chunk_text
        chunks = chunk_text("", uuid.uuid4(), "t1", "empty.txt")
        assert chunks == []

    def test_chunk_short_text_below_minimum(self):
        from app.services.chunker import chunk_text
        # Text with fewer than 10 tokens should be filtered out
        chunks = chunk_text("Hi.", uuid.uuid4(), "t1", "short.txt")
        assert len(chunks) == 0

    def test_chunk_ids_are_unique(self):
        from app.services.chunker import chunk_text
        text = "paragraph one. " * 100 + "\n\n" + "paragraph two. " * 100
        chunks = chunk_text(text, uuid.uuid4(), "t1", "doc.txt")
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids)), "chunk_ids must be unique"

    def test_chunk_indices_are_sequential(self):
        from app.services.chunker import chunk_text
        text = "word " * 2000
        chunks = chunk_text(text, uuid.uuid4(), "t1", "doc.txt")
        indices = [c["chunk_index"] for c in chunks]
        assert indices == list(range(len(indices)))


# ── OpenSearch hybrid fusion tests ────────────────────────────────────────────

class TestHybridFusion:
    """Test score normalisation and fusion without hitting OpenSearch."""

    def _normalise(self, hits: dict) -> dict:
        """Replicate the normalise logic from opensearch.py."""
        scores = {cid: h["_score"] for cid, h in hits.items()}
        mx = max(scores.values(), default=1.0)
        mn = min(scores.values(), default=0.0)
        rng = mx - mn or 1.0
        return {cid: (s - mn) / rng for cid, s in scores.items()}

    def test_normalise_single_item(self):
        hits = {"a": {"_score": 5.0}}
        norm = self._normalise(hits)
        # single item: (5-5)/(1) = 0 (range=0 so fallback to 1.0 → 0/1 = 0)
        assert norm["a"] == 0.0

    def test_normalise_multiple_items(self):
        hits = {"a": {"_score": 10.0}, "b": {"_score": 5.0}, "c": {"_score": 0.0}}
        norm = self._normalise(hits)
        assert norm["a"] == pytest.approx(1.0)
        assert norm["c"] == pytest.approx(0.0)
        assert 0.0 < norm["b"] < 1.0

    def test_fusion_weights_sum_to_one(self):
        from app.config import get_settings
        s = get_settings()
        assert round(s.bm25_weight + s.knn_weight, 2) == 1.0


# ── Settings validation ────────────────────────────────────────────────────────

class TestSettings:
    def test_invalid_weights_raise(self):
        from pydantic import ValidationError
        from app.config import Settings
        with pytest.raises(ValidationError):
            Settings(
                secret_key="x" * 32,
                bm25_weight=0.5,
                knn_weight=0.6,  # 0.5 + 0.6 = 1.1 → should fail
            )

    def test_valid_weights_pass(self):
        from app.config import Settings
        s = Settings(secret_key="x" * 32, bm25_weight=0.3, knn_weight=0.7)
        assert s.bm25_weight + s.knn_weight == pytest.approx(1.0)
