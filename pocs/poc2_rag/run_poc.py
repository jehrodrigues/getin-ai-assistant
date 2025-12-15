# pocs/poc2_rag/run_poc.py

import os
import sys
from pathlib import Path
from typing import List

from src.rag.loader import load_markdown_documents, chunk_documents
from src.rag.embeddings import embed_texts
from src.rag.store import build_vector_store
from src.rag.retrieval import retrieve
from src.rag.generation import generate_answer_from_chunks


CORPUS_DIR = Path("src/rag/corpus")

DEMO_QUERIES: List[str] = [
    "O restaurante oferece opções veganas?",
    "Tem espaço kids?",
    "Posso levar meu próprio vinho?",
    "O restaurante é bom para um jantar romântico?",
    "Quais métodos de pagamento o restaurante aceita?",
]


def build_store() -> tuple[list, list, object]:
    """
    Load the corpus, chunk the documents, embed the chunks and build the vector store.

    Returns:
        (docs, chunks, store)
    """
    print(f"Corpus directory: {CORPUS_DIR.resolve()}")

    # 1. Load documents
    docs = load_markdown_documents(CORPUS_DIR)
    print(f"Loaded {len(docs)} documents from corpus.")

    # 2. Chunk documents
    chunks = chunk_documents(docs, chunk_size=500, overlap=100)
    print(f"Generated {len(chunks)} chunks from documents.")

    if not chunks:
        raise RuntimeError("No chunks generated from corpus; aborting POC 2.")

    # 3. Embed all chunks
    texts = [c.text for c in chunks]
    print("Embedding chunks...")
    embeddings = embed_texts(texts)
    print(f"Embedded {len(embeddings)} chunks.")

    # 4. Build vector store
    store = build_vector_store(chunks, embeddings)
    print("Vector store built successfully.")

    return docs, chunks, store


def run_single_query(query: str, model_name: str, top_k: int = 3) -> None:
    """
    Run POC 2 end-to-end for a single query:
        - load corpus
        - chunk documents
        - embed chunks
        - build vector store
        - retrieve top-k chunks
        - generate answer with the configured model
    """
    print("=== POC 2 – Knowledge Layer (RAG) Demo ===")
    print(f"Model: {model_name}")
    docs, chunks, store = build_store()

    print("\n=== Query ===")
    print(query)

    # Retrieve relevant chunks
    results = retrieve(query, store, top_k=top_k)

    if not results:
        print("\nNo results found.")
        return

    # Generate answer using the configured model – only top-1 chunk for now
    answer = generate_answer_from_chunks(
        query,
        results[:1],
        model_name=model_name,
        temperature=0.1,
        max_tokens=256,
    )

    print(f"\n=== Resposta gerada ({model_name}) ===")
    print(answer)

    print(f"\n=== Top {len(results)} retrieved chunks (evidence) ===")
    for idx, r in enumerate(results, start=1):
        print(f"\n[{idx}] Score: {r.score:.3f}")
        print(f"Source: {r.source}")
        print("-" * 40)
        preview = r.text.strip()
        if len(preview) > 500:
            preview = preview[:500] + "..."
        print(preview)


def run_demo(model_name: str, top_k: int = 3) -> None:
    """
    Run multiple CEO-style demo queries end-to-end, with answer synthesis
    using the configured model.
    """
    print("=== POC 2 – Knowledge Layer (RAG) Demo ===")
    print(f"Model: {model_name}")
    docs, chunks, store = build_store()

    for query in DEMO_QUERIES:
        print("\n" + "=" * 80)
        print(f"Query: {query}")

        results = retrieve(query, store, top_k=top_k)

        if not results:
            print("No results found.")
            continue

        # Generate answer using the configured model
        answer = generate_answer_from_chunks(
            query,
            results[:1],          # keep prompt short & relevant
            model_name=model_name,
            temperature=0.1,
            max_tokens=256,
        )

        print(f"\n=== Resposta gerada ({model_name}) ===")
        print(answer)

        print(f"\n=== Top {len(results)} trechos usados como contexto ===")
        for idx, r in enumerate(results, start=1):
            print(f"\n  [{idx}] Score: {r.score:.3f}")
            print(f"  Source: {r.source}")
            print("  " + "-" * 40)
            preview = r.text.strip()
            if len(preview) > 400:
                preview = preview[:400] + "..."
            print("  " + preview.replace("\n", "\n  "))


def main() -> None:
    """
    Entry point for POC 2.

    Usage:
        # Demo mode (multiple queries, default model or env RAG_MODEL_NAME)
        python -m pocs.poc2_rag.run_poc

        # Single query (default model)
        python -m pocs.poc2_rag.run_poc "Minha pergunta em linguagem natural"
    """
    model_name = "gpt-4o"

    if len(sys.argv) >= 2:
        query = " ".join(sys.argv[1:])
        run_single_query(query=query, model_name=model_name, top_k=3)
    else:
        run_demo(model_name=model_name, top_k=3)


if __name__ == "__main__":
    main()