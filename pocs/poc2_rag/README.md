# POC 2 – Knowledge Layer (RAG) Validation

**Goal**

Validate that the application can build and query a **local Retrieval-Augmented Generation (RAG) layer**
to answer restaurant-related questions using unstructured documents.

Specifically, this PoC validates that the system can:

- load a small corpus of markdown documents from disk
- split documents into overlapping chunks suitable for semantic search
- generate vector embeddings using a local multilingual model
- store embeddings in an in-memory vector store
- retrieve the most relevant chunks for a natural language query
- return grounded answers based solely on retrieved context
- operate independently from GET IN APIs and reservation logic

This PoC establishes a **standalone knowledge layer** that is later
integrated into the agent workflows in PoCs 3 and 4.

**Example interactions**

```text
Usuário: O restaurante oferece opções veganas?
Resposta (RAG): Sim, o restaurante oferece opções veganas.

Usuário: Qual o horário de funcionamento?
Resposta (RAG): O restaurante funciona de segunda a sexta das 12h às 23h.
```
---

**How to run**

```bash
# from the project root
python -m pocs.poc2_rag.run_poc