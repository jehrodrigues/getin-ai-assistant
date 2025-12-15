# src/rag/retrieval.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

from src.rag.embeddings import embed_texts
from src.rag.store import VectorStore, SearchResult


@dataclass
class RetrievedChunk:
    """
    Higher-level representation of a retrieved chunk.

    Attributes:
        text: Chunk content.
        source: Original document path or identifier.
        score: Similarity score between query and chunk.
        chunk_id: Unique identifier of the chunk.
        doc_id: Identifier of the original document.
        index: Position of the chunk within the document.
    """
    text: str
    source: str
    score: float
    chunk_id: str
    doc_id: str
    index: int


def retrieve(
    query: str,
    store: VectorStore,
    top_k: int = 5,
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
) -> List[RetrievedChunk]:
    """
    Retrieve the most relevant chunks from the vector store for a given query.

    Args:
        query: Natural language query string.
        store: VectorStore instance containing chunks and embeddings.
        top_k: Maximum number of results to return.
        embed_fn: Optional embedding function. If not provided, `embed_texts`
                  from src.rag.embeddings is used.

    Returns:
        A list of RetrievedChunk instances sorted by relevance (descending).
    """
    if not query or not query.strip():
        raise ValueError("Query text is empty.")

    if embed_fn is None:
        effective_embed_fn = embed_texts
    else:
        effective_embed_fn = embed_fn

    # Embed the query
    query_vec = effective_embed_fn([query])[0]

    # Search the vector store
    results: List[SearchResult] = store.search(query_vec, top_k=top_k)

    retrieved: List[RetrievedChunk] = []
    for res in results:
        chunk = res.chunk
        retrieved.append(
            RetrievedChunk(
                text=chunk.text,
                source=chunk.source,
                score=res.score,
                chunk_id=chunk.id,
                doc_id=chunk.doc_id,
                index=chunk.index,
            )
        )

    return retrieved