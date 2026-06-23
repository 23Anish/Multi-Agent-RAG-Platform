"""
Chunking strategy:

1. Split by logical separators (paragraphs → sentences → words)
   using LangChain's RecursiveCharacterTextSplitter.
2. Each chunk is sized by *token count* (not characters) using
   tiktoken, which matches what the embedding model actually sees.
3. Overlap of 10% of chunk_size lets cross-boundary context survive
   retrieval without duplicating too much content.

Why this matters:
  - Chunks too large → embedding averages over too much, loses precision.
  - Chunks too small → lose sentence context, hurt BM25 quality.
  - 512 tokens ≈ sweet spot for Titan Embed v2.
"""
import re
import uuid

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 51  # ~10 %
ENCODING = tiktoken.get_encoding("cl100k_base")  # same family as Titan / GPT-4


def _count_tokens(text: str) -> int:
    return len(ENCODING.encode(text))


def _char_limit_for_tokens(target_tokens: int, sample: str = " ") -> int:
    """Approximate character limit for a given token budget."""
    # Average ~4 chars per token for English
    return target_tokens * 4


def chunk_text(
    text: str,
    document_id: uuid.UUID,
    tenant_id: str,
    filename: str,
    page_num: int | None = None,
) -> list[dict]:
    """
    Split `text` into overlapping token-bounded chunks.
    Returns a list of dicts ready for OpenSearch bulk indexing.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_char_limit_for_tokens(CHUNK_SIZE_TOKENS),
        chunk_overlap=_char_limit_for_tokens(CHUNK_OVERLAP_TOKENS),
        length_function=len,
        separators=["\n\n", "\n", ". ", "? ", "! ", ", ", " ", ""],
    )

    raw_chunks = splitter.split_text(text)

    chunks = []
    for idx, chunk_text in enumerate(raw_chunks):
        chunk_text = chunk_text.strip()
        if not chunk_text:
            continue

        token_count = _count_tokens(chunk_text)
        if token_count < 10:
            continue  # skip degenerate micro-chunks

        chunks.append(
            {
                "chunk_id": str(uuid.uuid4()),
                "document_id": str(document_id),
                "tenant_id": tenant_id,
                "filename": filename,
                "chunk_index": idx,
                "text": chunk_text,
                "token_count": token_count,
                "page_num": page_num,
                # `embedding` filled in by the Celery worker after embed call
            }
        )

    return chunks


def extract_text_from_pdf_bytes(data: bytes) -> list[tuple[int, str]]:
    """
    Extract (page_num, text) tuples from PDF bytes.
    Returns one entry per page.
    Uses PyMuPDF (fitz) for text extraction — handles most PDFs well.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise RuntimeError("PyMuPDF not installed — run: pip install pymupdf")

    doc = fitz.open(stream=data, filetype="pdf")
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if text:
            pages.append((page_num, text))
    doc.close()
    return pages
