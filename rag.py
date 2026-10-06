"""Retrieval layer: extract PDF text, split it into chunks, and find the chunks
most relevant to a question using BM25 keyword ranking."""

import re
from dataclasses import dataclass
from typing import BinaryIO

from pypdf import PdfReader
from rank_bm25 import BM25Plus


@dataclass
class Chunk:
    source: str  # file name
    page: int  # 1-indexed page number
    text: str


def extract_pages(file: BinaryIO, source: str) -> list[Chunk]:
    """Return one Chunk per non-empty PDF page."""
    reader = PdfReader(file)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(Chunk(source=source, page=i, text=text))
    return pages


def split_chunks(pages: list[Chunk], size: int = 1200, overlap: int = 200) -> list[Chunk]:
    """Split page text into overlapping character windows, keeping page numbers."""
    chunks = []
    step = size - overlap
    for p in pages:
        text = re.sub(r"\s+", " ", p.text)
        for start in range(0, max(len(text) - overlap, 1), step):
            piece = text[start : start + size].strip()
            if piece:
                chunks.append(Chunk(source=p.source, page=p.page, text=piece))
    return chunks


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class Retriever:
    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("No text found. Is the PDF scanned (image-only)?")
        self.chunks = chunks
        # BM25Plus keeps scores positive even for tiny documents with few chunks.
        self.index = BM25Plus([_tokenize(c.text) for c in chunks])

    def search(self, query: str, k: int = 6) -> list[Chunk]:
        scores = self.index.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.chunks[i] for i in ranked[:k]]


def format_context(chunks: list[Chunk]) -> str:
    """Render retrieved chunks as tagged sources the model can cite."""
    return "\n\n".join(
        f'<source file="{c.source}" page="{c.page}">\n{c.text}\n</source>' for c in chunks
    )
