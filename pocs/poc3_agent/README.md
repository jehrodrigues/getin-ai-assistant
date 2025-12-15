# POC 3 – Single-turn Agent Workflow (LangGraph + GET IN + RAG)

**Goal**

Validate that the application can execute a **single-turn conversational agent workflow**
that connects LLM reasoning with real GET IN APIs and a RAG-based FAQ module.

Specifically, this PoC validates that the system can:

- classify high-level user intent using an LLM  
  (`check_availability`, `create_reservation`, `view_next_reservation`,
  `cancel_reservation`, `restaurant_faq`)
- extract structured parameters from free-text user input (date, time, party size, etc.)
- execute the correct backend action based on intent
- integrate with GET IN APIs to:
  - check availability via `/schedules/units/:unit_id`
  - create reservations via `POST /reservations`
  - consult upcoming reservations via `/reservations/next`
  - list and cancel reservations via `/reservations`
- retrieve restaurant FAQs using a local **RAG (Retrieval-Augmented Generation)** pipeline
- transform structured API responses into natural PT-BR answers using an LLM
- handle API and validation errors gracefully and explain them to the user

This PoC establishes the **core agent architecture** that later evolves into
a multi-turn conversational flow in PoC 4.

**Example interactions**

```text
Usuário: Tem mesa para 4 amanhã às 12:00?
Assistente: Temos disponibilidade no Salão Principal. Gostaria de reservar?

Usuário: Quero cancelar minha reserva de amanhã às 20h.
Assistente: Não encontrei nenhuma reserva com essas informações.
```
---

**How to run**

```bash
# from the project root
python -m pocs.poc3_agent.run_poc
POC4_DEBUG=1 python -m pocs.poc3_agent.run_poc