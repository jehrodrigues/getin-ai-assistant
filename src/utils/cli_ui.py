# src/utils/cli_ui.py
from __future__ import annotations

import os
from typing import Any, Dict, Optional


def _mode() -> str:
    """
    POC_MODE:
      - unset / normal : modo normal
      - 1 / true / yes : modo screenshot (aesthetic)
      - screenshot     : modo screenshot (legado)
    """
    raw = (os.getenv("POC_MODE") or "").strip().lower()

    if raw in {"1", "true", "yes", "on", "screenshot"}:
        return "screenshot"

    return "normal"


def _emoji_enabled() -> bool:
    env = os.getenv("POC_EMOJI")
    if env is not None:
        return env.strip() in {"1", "true", "yes", "on"}
    return _mode() == "screenshot"


def banner(title: str) -> None:
    """
    Banner só aparece fora do modo screenshot.
    """
    if _mode() == "screenshot":
        return
    print("=" * 60)
    print(f" {title} ")
    print("=" * 60)


def divider() -> None:
    """
    Separador:
      - screenshot: uma linha em branco (mais clean)
      - normal/debug: linha contínua
    """
    if _mode() == "screenshot":
        print()
        return
    print("-" * 60)


def print_user(text: str) -> None:
    """
    Prints a nice user message block (better for screenshot).
    """
    if _emoji_enabled():
        print(f"🧑 Você: {text}")
    else:
        print(f"Você: {text}")


def print_bot(text: str) -> None:
    """
    Prints a nice assistant message block (better for screenshot).
    """
    if _emoji_enabled():
        print(f"🤖 Assistente: {text}")
    else:
        print(f"Assistente: {text}")


def should_hide_debug_dump() -> bool:
    """
    When POC4_DEBUG=1, we print full state dumps.
    In screenshot mode, hide dumps by default.
    """
    return _mode() == "screenshot"


def mask_secret(value: Optional[str], keep_last: int = 4) -> str:
    if not value:
        return "<empty>"
    if len(value) <= keep_last:
        return "*" * len(value)
    return "*" * (len(value) - keep_last) + value[-keep_last:]


def summarize_action_result(action_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep this helper in case you want to use it later,
    but the current run_poc4 will NOT print it.
    """
    if not isinstance(action_result, dict):
        return {"title": "Ação executada.", "details": None, "kind": "info"}

    t = action_result.get("type") or "unknown"
    ok = action_result.get("ok")

    if ok is True:
        if t == "availability":
            return {"title": "Disponibilidade consultada.", "details": None, "kind": "ok"}
        if t == "create_reservation":
            return {"title": "Reserva criada.", "details": None, "kind": "ok"}
        if t in {"view_next_reservation", "list_reservations"}:
            return {"title": "Reservas consultadas.", "details": None, "kind": "ok"}
        return {"title": "Ação executada com sucesso.", "details": None, "kind": "ok"}

    if ok is False:
        err = (action_result.get("error") or {}).get("message") or "Erro na ação."
        return {"title": "Falha ao executar ação.", "details": err, "kind": "error"}

    return {"title": "Ação executada.", "details": None, "kind": "info"}