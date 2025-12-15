# src/agent/actions/rag_action.py

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.rag.retrieval import RetrievedChunk, retrieve
from src.rag.store import VectorStore, get_default_store
from src.rag.generation import generate_answer_from_chunks

_store: Optional[VectorStore] = None


def _get_store() -> VectorStore:
    """
    Lazily obtain a VectorStore instance for RAG.

    Uses `get_default_store()` from src.rag.store, which builds and caches
    a vector store from the markdown corpus directory.
    """
    global _store
    if _store is not None:
        return _store

    _store = get_default_store()
    return _store


def _retrieve_chunks_for_query(query: str, top_k: int = 3) -> List[RetrievedChunk]:
    """
    Thin wrapper around the POC 2 retrieval logic.

    Uses:
        retrieve(query, store, top_k=top_k)
    """
    store = _get_store()
    return retrieve(query=query, store=store, top_k=top_k)


def answer_with_rag(user_input: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Use the POC 2 RAG pipeline to answer a general restaurant FAQ.

    Intended for the `restaurant_faq` intent:

      1. Use the user message as the RAG query.
      2. Retrieve relevant documentation chunks from the vector store.
      3. Call `generate_answer_from_chunks` to produce a short answer in PT-BR.
      4. Return a structured payload with the final answer.

    Parameters
    ----------
    user_input : str
        Original user message in PT-BR.
    params : Dict[str, Any]
        Structured parameters from the extractor. Not strictly required here,
        but kept for symmetry and future refinements.

    Returns
    -------
    Dict[str, Any]
        Success example:

        {
          "type": "restaurant_faq",
          "ok": true,
          "request": { ... },
          "answer": "Resposta em PT-BR...",
          "meta": {
            "used_rag": true,
            "chunks_count": 3
          }
        }

        Failure example:

        {
          "type": "restaurant_faq",
          "ok": false,
          "error": {
            "code": "RAG_RETRIEVAL_ERROR" | "RAG_GENERATION_ERROR",
            "message": "Mensagem em PT-BR amigável.",
            "details": { ... }
          },
          "request": { ... }
        }
    """
    request_payload: Dict[str, Any] = {
        "user_input": user_input,
        "params": params,
    }

    query = user_input

    try:
        chunks = _retrieve_chunks_for_query(query, top_k=3)
    except Exception as exc:
        return {
            "type": "restaurant_faq",
            "ok": False,
            "error": {
                "code": "RAG_RETRIEVAL_ERROR",
                "message": (
                    "Não consegui acessar a documentação interna para responder sua pergunta."
                ),
                "details": {
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                },
            },
            "request": request_payload,
        }

    try:
        answer = generate_answer_from_chunks(
            query=query,
            chunks=chunks,
        )
    except Exception as exc:
        return {
            "type": "restaurant_faq",
            "ok": False,
            "error": {
                "code": "RAG_GENERATION_ERROR",
                "message": (
                    "Ocorreu um erro ao gerar a resposta baseada na documentação interna."
                ),
                "details": {
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                },
            },
            "request": request_payload,
        }

    return {
        "type": "restaurant_faq",
        "ok": True,
        "request": request_payload,
        "answer": answer,
        "meta": {
            "used_rag": True,
            "chunks_count": len(chunks),
        },
    }