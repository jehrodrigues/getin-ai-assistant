# src/rag/loader.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass
class Document:
    """Represents a full source document before chunking."""
    id: str
    text: str
    source: str  # usually the filename


@dataclass
class Chunk:
    """
    Represents a chunk of text derived from a source document.

    Attributes:
        id: Globally unique identifier for this chunk.
        doc_id: Identifier of the original document.
        index: Position of the chunk within the document (0-based).
        text: Chunk content.
        source: Original source (e.g. filename).
    """
    id: str
    doc_id: str
    index: int
    text: str
    source: str


def load_markdown_documents(corpus_dir: str | Path) -> List[Document]:
    """
    Load all .md files from the given corpus directory as Document objects.

    Args:
        corpus_dir: Path to the directory containing markdown files.

    Returns:
        A list of Document instances, one per markdown file.
    """
    corpus_path = Path(corpus_dir)
    if not corpus_path.exists() or not corpus_path.is_dir():
        raise ValueError(f"Corpus directory does not exist or is not a directory: {corpus_dir}")

    documents: List[Document] = []

    for md_path in sorted(corpus_path.glob("*.md")):
        text = md_path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        doc_id = md_path.stem
        documents.append(
            Document(
                id=doc_id,
                text=text,
                source=str(md_path),
            )
        )

    return documents


def _split_text_into_chunks(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> List[str]:
    """
    Split a long string into overlapping character-based chunks.

    This is a simple, model-agnostic splitter suitable for POC usage.
    You can later replace it with a token-based splitter if needed.

    Args:
        text: Input text to be chunked.
        chunk_size: Target size of each chunk (in characters).
        overlap: Number of characters to overlap between consecutive chunks.

    Returns:
        A list of text chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if overlap < 0:
        raise ValueError("overlap cannot be negative.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    text = text.strip()
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == length:
            break
        start = end - overlap

    return chunks


def chunk_documents(
    documents: Iterable[Document],
    chunk_size: int = 500,
    overlap: int = 100,
) -> List[Chunk]:
    """
    Transform a list of Document objects into a list of Chunk objects.

    Args:
        documents: Iterable of Document instances to be chunked.
        chunk_size: Target size for each chunk (in characters).
        overlap: Overlap size between consecutive chunks.

    Returns:
        A flat list of Chunk instances for all documents.
    """
    all_chunks: List[Chunk] = []

    for doc in documents:
        raw_chunks = _split_text_into_chunks(
            text=doc.text,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for idx, chunk_text in enumerate(raw_chunks):
            chunk_id = f"{doc.id}::chunk-{idx}"
            all_chunks.append(
                Chunk(
                    id=chunk_id,
                    doc_id=doc.id,
                    index=idx,
                    text=chunk_text,
                    source=doc.source,
                )
            )

    return all_chunks