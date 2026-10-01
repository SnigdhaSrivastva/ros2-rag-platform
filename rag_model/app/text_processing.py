"""Pure text-processing helpers shared by the ETL, featurization and API layers.

Kept free of heavy dependencies (ML models, database clients) so it can be
unit-tested quickly in CI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping
from urllib.parse import urldefrag, urljoin, urlparse


@dataclass
class Chunk:
    text: str
    chunk_id: int


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[Chunk]:
    """Split text into overlapping, whitespace-normalized chunks."""
    if chunk_size <= 0:
        raise ValueError('chunk_size must be positive')
    if not 0 <= overlap < chunk_size:
        raise ValueError('overlap must be >= 0 and smaller than chunk_size')

    text = ' '.join(text.split())
    if not text:
        return []
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(Chunk(text=text[start:end], chunk_id=idx))
        if end == len(text):
            break
        start = end - overlap
        idx += 1
    return chunks


def build_prompt(question: str, contexts: list[str]) -> str:
    context = '\n\n'.join(contexts)
    return (
        'You are a production ROS2 expert assistant. '
        'Answer the question using ONLY the provided context. '
        'If context is insufficient, say what is missing.\n\n'
        f'Question: {question}\n\n'
        f'Context:\n{context}\n\n'
        'Return a concise, implementation-focused answer with bullet points.'
    )


def collect_contexts(payloads: Iterable[Mapping[str, Any] | None]) -> tuple[list[str], list[str]]:
    """Extract non-empty context texts and their source labels from search-hit payloads."""
    contexts: list[str] = []
    sources: list[str] = []
    for payload in payloads:
        payload = payload or {}
        text = (payload.get('text') or '').strip()
        if text:
            contexts.append(text)
            sources.append(payload.get('url') or payload.get('source') or 'unknown')
    return contexts, sources


def same_domain_links(page_url: str, hrefs: Iterable[str], domain: str) -> list[str]:
    """Resolve hrefs against page_url, keep only http(s) links on `domain`.

    Fragments are stripped so `page#a` and `page#b` are crawled once, and
    duplicates are removed while preserving order.
    """
    links: list[str] = []
    seen: set[str] = set()
    for href in hrefs:
        absolute, _ = urldefrag(urljoin(page_url, href))
        parsed = urlparse(absolute)
        if parsed.scheme not in ('http', 'https') or parsed.netloc != domain:
            continue
        if absolute not in seen:
            seen.add(absolute)
            links.append(absolute)
    return links
