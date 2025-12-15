# src/agent/graph/state.py

from __future__ import annotations

from typing import Any, Dict, Optional, TypedDict


class AgentState(TypedDict, total=False):
    """
    Shared state object that flows through the LangGraph workflow.

    Fields
    ------
    user_input : str
        Raw user message in PT-BR for the current turn.

    intent : Optional[str]
        High-level intent label predicted by the intent classifier.
        Examples: "check_availability", "create_reservation",
                  "view_next_reservation", "list_reservations",
                  "cancel_reservation", "restaurant_faq", "other".

    params : Dict[str, Any]
        Structured parameters extracted from the user message.
        Examples: date, time, party_size, phone, email, etc.

    action_result : Optional[Dict[str, Any]]
        Result returned by the action layer (API/RAG).
        It should follow the standardized shape used by the actions:
            {
              "type": "...",
              "ok": True/False,
              "request": {...},
              "response": {...} | None,
              "error": {...} | None,
              ...
            }

    answer : Optional[str]
        Final natural language answer in PT-BR that will be shown to the user.
    """

    user_input: str
    intent: Optional[str]
    params: Dict[str, Any]
    action_result: Optional[Dict[str, Any]]
    answer: Optional[str]