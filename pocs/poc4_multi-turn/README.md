# POC 4 – Multi-turn Reservation Agent (LangGraph + GET IN)

**Goal**

Validate that the application can support a **full, multi-turn reservation flow**
for a single restaurant using conversational AI.

Specifically, this PoC validates that the system can:

- maintain conversational state across multiple user turns using **LangGraph**
- classify user intent and incrementally extract structured parameters with LLMs
- perform slot-filling when required reservation data is missing
- check table availability using the GET IN endpoint  
  `/schedules/units/:unit_id`
- present available sectors using user-friendly sector names
- create real reservations via `POST /reservations`
- handle and recover from business-rule errors returned by the GET IN API
  (e.g. duplicate reservations for the same phone, date and time)
- generate coherent PT-BR responses grounded in API results and conversation context

This PoC demonstrates an **end-to-end conversational booking experience**,
from the initial availability query to successful reservation creation,
and is intended to validate architectural and interaction design choices
during the **Discovery Phase**.

**Example interactions**

```text
Usuário: Tem mesa para 4 amanhã às 12:00?
Assistente: Perfeito! Temos disponibilidade para 4 pessoas amanhã às 12:00 no Salão Principal.
Gostaria de reservar uma mesa nesse setor?

Usuário: Sim, pode reservar.
Assistente: Ótimo! Para concluir sua reserva, preciso do seu nome, telefone e e-mail.

Usuário: Meu nome é Jessica, telefone 11999999999 e e-mail jessica@example.com
Assistente: Perfeito, Jessica! Sua reserva foi criada com sucesso para 13/12/2025 às 12:00,
para 4 pessoas no Salão Principal.
O código da sua reserva é ABC123.

Usuário: Quero cancelar minha reserva de amanhã às 10:00.
Assistente: Localizei sua reserva para amanhã às 10:00.
Posso confirmar o cancelamento?

Usuário: Sim, pode cancelar.
Assistente: Sua reserva foi cancelada com sucesso. Se precisar de algo mais, estou à disposição.

Usuário: Tenho alguma reserva para hoje?
Assistente: Encontrei uma reserva em seu nome hoje às 19:30 para 2 pessoas no Salão Principal.
Deseja mais detalhes ou alguma alteração?
```
---

**How to run**

```bash
# from the project root
python -m pocs.poc4_multi-turn.run_poc
POC4_DEBUG=1 python -m pocs.poc4_multi-turn.run_poc