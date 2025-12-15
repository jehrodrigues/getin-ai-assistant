# src/agent/extractors/time_utils.py

from __future__ import annotations

import re
from datetime import datetime, date, timedelta
from typing import Optional


def _today() -> date:
    """
    Wrapper around date.today() to make testing easier if needed.
    """
    return date.today()


def normalize_date(raw: str) -> str:
    """
    Normalize a PT-BR date expression into ISO format (YYYY-MM-DD).

    Supported patterns:
      - Natural language (PT-BR):
          * "hoje"
          * "amanhã", "amanha"
          * "depois de amanhã", "depois de amanha"
      - Explicit ISO:
          * "2025-03-10"
      - Brazilian format:
          * "10/03/2025"

    Parameters
    ----------
    raw : str
        Original date string extracted by the LLM.

    Returns
    -------
    str
        Normalized date in ISO format: "YYYY-MM-DD".

    Raises
    ------
    ValueError
        If the input cannot be interpreted as a supported date.
    """
    if not raw:
        raise ValueError("Empty date string.")

    text = raw.strip().lower()

    # Natural PT-BR keywords
    today = _today()
    if text == "hoje":
        return today.isoformat()

    if text in {"amanha", "amanhã"}:
        return (today + timedelta(days=1)).isoformat()

    if text in {"depois de amanha", "depois de amanhã"}:
        return (today + timedelta(days=2)).isoformat()

    # ISO pattern: YYYY-MM-DD
    iso_match = re.fullmatch(r"\d{4}-\d{2}-\d{2}", text)
    if iso_match:
        # Basic sanity check via datetime
        dt = datetime.strptime(text, "%Y-%m-%d")
        return dt.date().isoformat()

    # Brazilian pattern: DD/MM/YYYY
    br_match = re.fullmatch(r"(\d{2})/(\d{2})/(\d{4})", text)
    if br_match:
        day, month, year = br_match.groups()
        dt = datetime.strptime(f"{day}/{month}/{year}", "%d/%m/%Y")
        return dt.date().isoformat()

    raise ValueError(f"Unsupported date format: {raw!r}")


def normalize_time(raw: str) -> str:
    """
    Normalize a PT-BR time expression into HH:MM (24h) format.

    Supported patterns:
      - Explicit times:
          * "20h", "20 h", "20hs", "20 hs"
          * "20:00", "20", "19:30"
          * "às 19h", "as 19h", "a 19h"
      - Coarse expressions (mapped heuristically):
          * "no almoço", "almoço", "almoco"   -> "12:00"
          * "no jantar", "jantar"             -> "20:00"
          * "de manhã", "manhã", "manha"      -> "09:00"
          * "à tarde", "a tarde", "tarde"     -> "15:00"
          * "à noite", "a noite", "noite"     -> "20:00"
    """
    if not raw:
        raise ValueError("Empty time string.")

    text = raw.strip().lower()

    # Coarse PT-BR expressions
    coarse_map = {
        "almoço": "12:00",
        "almoco": "12:00",
        "no almoço": "12:00",
        "no almoco": "12:00",
        "jantar": "20:00",
        "no jantar": "20:00",
        "de manhã": "09:00",
        "de manha": "09:00",
        "manhã": "09:00",
        "manha": "09:00",
        "à tarde": "15:00",
        "a tarde": "15:00",
        "tarde": "15:00",
        "à noite": "20:00",
        "a noite": "20:00",
        "noite": "20:00",
    }
    if text in coarse_map:
        return coarse_map[text]

    # Remove common prefixes like "às", "as", "a"
    for prefix in ("às ", "as ", "a "):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    # Remove caracteres estranhos (ponto, vírgula, interrogação, etc.),
    # mantendo apenas dígitos, 'h' e ':'
    cleaned = []
    for ch in text:
        if ch.isdigit() or ch in {"h", ":"}:
            cleaned.append(ch)
    text = "".join(cleaned)

    # Normalizar variantes: "20 hs", "20hs" -> "20h"
    text = text.replace("hs", "h")

    # Trocar "h" por ":" (ex.: "20h30" -> "20:30", "20h" -> "20:")
    text = text.replace("h", ":")

    # Agora esperamos algo como "20", "20:", "20:30"
    match = re.fullmatch(r"(\d{1,2})(?::(\d{1,2}))?", text)
    if not match:
        raise ValueError(f"Unsupported time format: {raw!r}")

    hour_str, minute_str = match.groups()
    hour = int(hour_str)
    minute = int(minute_str) if minute_str is not None else 0

    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"Invalid time value: {raw!r}")

    return f"{hour:02d}:{minute:02d}"


def combine_to_iso(date_str: str, time_str: str) -> str:
    """
    Combine a normalized date (YYYY-MM-DD) and time (HH:MM) into
    an ISO 8601 datetime string (YYYY-MM-DDTHH:MM:00).

    Parameters
    ----------
    date_str : str
        Date in ISO format ("YYYY-MM-DD").
    time_str : str
        Time in "HH:MM" format.

    Returns
    -------
    str
        ISO datetime string without timezone: "YYYY-MM-DDTHH:MM:00".

    Raises
    ------
    ValueError
        If the inputs do not conform to the expected formats.
    """
    # Basic validation via datetime parsing
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
    except Exception as exc:
        raise ValueError(f"Invalid ISO date: {date_str!r}") from exc

    try:
        t = datetime.strptime(time_str, "%H:%M").time()
    except Exception as exc:
        raise ValueError(f"Invalid time: {time_str!r}") from exc

    combined = datetime.combine(d, t)
    # Truncate seconds to ":00"
    return combined.replace(second=0, microsecond=0).isoformat()