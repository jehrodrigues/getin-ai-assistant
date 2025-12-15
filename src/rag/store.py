# src/rag/store.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional

from pathlib import Path

from src.rag.loader import Chunk, load_markdown_documents, chunk_documents
from src.rag.embeddings import embed_texts


def _cosine_similarity(vec1: Sequence[float], vec2: Sequence[float]) -> float:
    """
    Compute the cosine similarity between two vectors using pure Python.

    Args:
        vec1: First vector.
        vec2: Second vector.

    Returns:
        Cosine similarity in the range [-1, 1]. If any vector has zero norm,
        the similarity is defined as 0.0.
    """
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have the same dimension.")

    dot = 0.0
    norm1 = 0.0
    norm2 = 0.0

    for a, b in zip(vec1, vec2):
        dot += a * b
        norm1 += a * a
        norm2 += b * b

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    return dot / (norm1**0.5 * norm2**0.5)


@dataclass
class SearchResult:
    """
    Represents a single search result from the vector store.

    Attributes:
        chunk: The retrieved Chunk.
        score: Cosine similarity score between query and chunk embedding.
    """
    chunk: Chunk
    score: float


@dataclass
class VectorStore:
    """
    In-memory vector store for POC 2.

    Stores embeddings and their associated chunks, and supports
    cosine-similarity search.
    """
    chunks: List[Chunk]
    vectors: List[List[float]]
    dimension: int

    def __post_init__(self) -> None:
        if len(self.chunks) != len(self.vectors):
            raise ValueError(
                f"Number of chunks ({len(self.chunks)}) does not match "
                f"number of vectors ({len(self.vectors)})."
            )
        if self.vectors:
            dim = len(self.vectors[0])
            if any(len(v) != dim for v in self.vectors):
                raise ValueError("All vectors must have the same dimension.")
            if dim != self.dimension:
                raise ValueError(
                    f"Declared dimension ({self.dimension}) does not match "
                    f"actual vector dimension ({dim})."
                )

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int = 5,
    ) -> List[SearchResult]:
        """
        Search the vector store for the most similar chunks to a query vector.

        Args:
            query_vector: Embedding vector for the query.
            top_k: Maximum number of results to return.

        Returns:
            A list of SearchResult objects sorted by similarity (descending).
        """
        if not self.chunks or not self.vectors:
            return []

        if len(query_vector) != self.dimension:
            raise ValueError(
                f"Query vector dimension ({len(query_vector)}) does not match "
                f"store dimension ({self.dimension})."
            )

        scored: List[Tuple[Chunk, float]] = []

        for chunk, vec in zip(self.chunks, self.vectors):
            score = _cosine_similarity(query_vector, vec)
            scored.append((chunk, score))

        # Sort by score descending
        scored.sort(key=lambda pair: pair[1], reverse=True)

        results: List[SearchResult] = []
        for chunk, score in scored[:top_k]:
            results.append(SearchResult(chunk=chunk, score=score))

        return results


def build_vector_store(
    chunks: List[Chunk],
    embeddings: List[List[float]],
) -> VectorStore:
    """
    Build a VectorStore from a list of chunks and their corresponding embeddings.

    Args:
        chunks: List of Chunk instances.
        embeddings: List of embedding vectors, same order and length as chunks.

    Returns:
        A VectorStore instance ready for search operations.
    """
    if not chunks:
        raise ValueError("Cannot build a vector store with no chunks.")
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Number of chunks ({len(chunks)}) does not match number of embeddings ({len(embeddings)})."
        )

    dimension = len(embeddings[0])
    return VectorStore(chunks=chunks, vectors=embeddings, dimension=dimension)


_default_store: Optional[VectorStore] = None


def get_default_store(corpus_dir: str | Path = "src/rag/corpus") -> VectorStore:
    """
    Lazily build and cache a VectorStore from the markdown corpus.

    This is intended for POC 2 / POC 3 usage, where a single in-memory store
    is sufficient for all queries.

    Args:
        corpus_dir: Directory containing the .md documentation files.

    Returns
    -------
    VectorStore
        An in-memory vector store ready for similarity search.

    Raises
    ------
    ValueError
        If the corpus directory is invalid or empty.
    """
    global _default_store
    if _default_store is not None:
        return _default_store

    corpus_path = Path(corpus_dir)
    documents = load_markdown_documents(corpus_path)
    chunks = chunk_documents(documents)

    # Compute embeddings for each chunk text
    embeddings = embed_texts([c.text for c in chunks])

    _default_store = build_vector_store(chunks, embeddings)
    return _default_store