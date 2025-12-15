# pocs/poc3_agent/run_poc.py

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict

from src.agent.graph.workflow import run_agent


def _print_header() -> None:
    """
    Print a simple header for the POC 3 demo.
    """
    print("=" * 60)
    print(" POC 3 – Agent Workflow (LangGraph + GET IN + RAG) ")
    print("=" * 60)


def _pretty_print_debug(state: Dict[str, Any]) -> None:
    """
    Optional debug print of internal state (intent, params, action_result).

    Enabled when the environment variable POC3_DEBUG is set to a non-empty value.
    """
    if not os.getenv("POC3_DEBUG"):
        return

    intent = state.get("intent")
    params = state.get("params")
    action_result = state.get("action_result")

    print("\n[DEBUG] Intent:", intent)
    print("[DEBUG] Params:")
    print(json.dumps(params, ensure_ascii=False, indent=2))

    print("[DEBUG] Action result (truncated):")
    # Avoid printing extremely large payloads; truncate if necessary
    try:
        serialized = json.dumps(action_result, ensure_ascii=False, indent=2)
    except TypeError:
        serialized = str(action_result)

    max_len = 2000
    if len(serialized) > max_len:
        print(serialized[:max_len] + "\n... [truncated]")
    else:
        print(serialized)
    print("-" * 60)


def _run_single_turn(user_input: str) -> None:
    """
    Run the agent for a single user input and print the final answer.
    """
    state = run_agent(user_input)
    _print_header()
    print(f"Usuário: {user_input}\n")

    _pretty_print_debug(state)

    answer = state.get("answer") or "Não consegui gerar uma resposta no momento."
    print("Assistente:", answer)
    print()


def _interactive_loop() -> None:
    """
    Start an interactive loop for manual testing of the agent.
    """
    _print_header()
    print("Digite a mensagem para o assistente (PT-BR).")
    print("Para sair, digite: sair | exit | quit\n")

    while True:
        try:
            user_input = input("Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando POC 3. Até mais!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"sair", "exit", "quit"}:
            print("Encerrando POC 3. Até mais!")
            break

        state = run_agent(user_input)
        _pretty_print_debug(state)

        answer = state.get("answer") or "Não consegui gerar uma resposta no momento."
        print("Assistente:", answer)
        print()


def main() -> None:
    """
    Entry point for the POC 3 script.

    Usage:
        # One-shot mode:
        python -m pocs.poc3_agent.run_poc3 "Quero fazer uma reserva hoje às 20h para 2 pessoas"

        # Interactive mode:
        python -m pocs.poc3_agent.run_poc3
    """
    if len(sys.argv) > 1:
        # One-shot mode: join all CLI args as a single user message
        user_input = " ".join(sys.argv[1:]).strip()
        if not user_input:
            print("Nenhuma mensagem fornecida na linha de comando.")
            sys.exit(1)
        _run_single_turn(user_input)
    else:
        # Interactive mode
        _interactive_loop()


if __name__ == "__main__":
    main()