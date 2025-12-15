# pocs/poc4_multi-turn/run_poc.py

from __future__ import annotations

import json
import os
from typing import Any, Dict

from src.agent.graph.workflow import build_workflow
from src.agent.graph.state import AgentState
from src.utils.cli_ui import (
    banner,
    divider,
    print_user,
    print_bot,
    should_hide_debug_dump,
)

POC_MODE = os.getenv("POC_MODE", "").strip() in {"1", "true", "True", "yes", "on"}


def _print_header() -> None:
    banner("POC 4 – Fluxo de Reserva Fim a Fim (GET IN API)")


def _pretty_print_debug(state: Dict[str, Any]) -> None:
    if not os.getenv("POC4_DEBUG"):
        return

    if should_hide_debug_dump():
        return

    print("\n[DEBUG] STATE:")
    debug_state = {
        "intent": state.get("intent"),
        "params": state.get("params"),
        "action_result": state.get("action_result"),
    }
    try:
        print(json.dumps(debug_state, ensure_ascii=False, indent=2))
    except TypeError:
        print(debug_state)
    print("-" * 60)


def interactive_reservation_loop() -> None:
    """
    Interactive loop that keeps AgentState between turns.
    """
    app = build_workflow()

    state: AgentState = {
        "user_input": "",
        "intent": None,
        "params": {},
        "action_result": None,
        "answer": None,
    }

    _print_header()

    if not POC_MODE:
        print("Você está falando com o assistente de reservas (POC 4).")
        print("Descreva a reserva que deseja fazer, por exemplo:")
        print('  "Quero fazer uma reserva amanhã às 20h para 2 pessoas."')
        print("Para sair, digite: sair | exit | quit\n")

    while True:
        try:
            user_input = input("> " if POC_MODE else "Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando POC 4. Até mais!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"sair", "exit", "quit"}:
            print("\nEncerrando POC 4. Até mais!")
            break

        print_user(user_input)

        state["user_input"] = user_input
        state = app.invoke(state)

        _pretty_print_debug(state)

        answer = state.get("answer") or "Não consegui gerar uma resposta no momento."
        print_bot(answer)

        divider()


def main() -> None:
    interactive_reservation_loop()


if __name__ == "__main__":
    main()