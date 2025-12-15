# src/rag/embeddings.py

from __future__ import annotations

from typing import List, Sequence, Optional

from sentence_transformers import SentenceTransformer


# Default local embedding model for POC 2.
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

_model: Optional[SentenceTransformer] = None


def _get_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    """
    Lazily load and cache the sentence-transformers model.

    The first call will download the model if it is not available locally.
    Subsequent calls reuse the same instance.
    """
    global _model
    if _model is not None:
        return _model

    _model = SentenceTransformer(model_name)
    return _model


def embed_texts(
    texts: Sequence[str],
    model: str | None = None,
    batch_size: int = 32,
) -> List[List[float]]:
    """
    Generate embeddings for a list of texts using a local sentence-transformers model.

    Args:
        texts: Sequence of strings to embed.
        model: Optional model name. If None, DEFAULT_EMBEDDING_MODEL is used.
        batch_size: Batch size for model inference.

    Returns:
        A list of embedding vectors, one per input text, in the same order.

    Raises:
        ValueError: If no texts are provided.
    """
    if not texts:
        raise ValueError("No texts provided for embedding.")

    model_name = model or DEFAULT_EMBEDDING_MODEL
    st_model = _get_model(model_name)

    # Normalise and preserve indexing
    cleaned_texts: List[str] = [t if t is not None else "" for t in texts]
    if all(not t.strip() for t in cleaned_texts):
        raise ValueError("All texts are empty or whitespace; nothing to embed.")

    # sentence-transformers returns a numpy array; convert to list[list[float]]
    embeddings_array = st_model.encode(
        cleaned_texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embeddings_array.tolist()