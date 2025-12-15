# src/agent/extractors/params_extractor.py

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from src.agent.llm_utils import call_llm, build_single_turn_prompt

SYSTEM_PROMPT = """
Você é responsável por extrair parâmetros estruturados a partir de uma mensagem de usuário
para um assistente de restaurante.

A mensagem sempre estará em português do Brasil.

Você deve analisar o texto e retornar um JSON com os seguintes campos:

{
  "date": string ou null,          // data mencionada (ex: "hoje", "amanhã", "2024-03-10")
  "time": string ou null,          // horário mencionado (ex: "20h", "19:30", "no almoço")
  "party_size": número ou null,    // quantidade de pessoas (ex: 2, 4, 10)
  "name": string ou null,          // nome da pessoa, se mencionado
  "phone": string ou null,         // telefone ou celular, se mencionado
  "email": string ou null,         // e-mail, se mencionado
  "notes": string ou null          // qualquer informação adicional relevante (ex: "mesa na janela", "aniversário")
}

Regras importantes:
- Se um campo não estiver claro na mensagem, use null.
- Não invente dados; apenas preencha o que estiver explícito ou muito óbvio.
- O JSON deve ser VÁLIDO, sem comentários, sem vírgulas sobrando.
- NÃO escreva nada antes ou depois do JSON.
Apenas retorne o JSON.
""".strip()


def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Try to extract a JSON object from the LLM output.

    The model *should* return only a JSON object, but this helper is defensive
    and tries to parse the first {...} block if extra text is present.
    """
    if not text:
        return None

    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except Exception:
        return None


def _ensure_param_schema(data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Ensure that the returned dictionary always has the expected keys.

    Unknown/invalid values are coerced to None where appropriate.
    """
    if data is None:
        data = {}

    date = data.get("date")
    time = data.get("time")
    party_size = data.get("party_size")
    name = data.get("name")
    phone = data.get("phone")
    email = data.get("email")
    notes = data.get("notes")

    if not isinstance(date, str):
        date = None
    if not isinstance(time, str):
        time = None

    if isinstance(party_size, str):
        try:
            party_size = int(party_size)
        except ValueError:
            party_size = None
    if not isinstance(party_size, (int, float)):
        party_size = None

    if not isinstance(name, str):
        name = None
    if not isinstance(phone, str):
        phone = None
    if not isinstance(email, str):
        email = None
    if not isinstance(notes, str):
        notes = None

    normalized: Dict[str, Any] = {
        "date": date,
        "time": time,
        "party_size": int(party_size) if isinstance(party_size, (int, float)) else None,
        "name": name,
        "phone": phone,
        "email": email,
        "notes": notes,
        "raw": data,
    }

    _maybe_postprocess_datetime(normalized)

    return normalized


def _maybe_postprocess_datetime(params: Dict[str, Any]) -> None:
    """
    Optionally post-process date/time using functions from time_utils, if present.

    This function is defensive:
    - It does not fail if time_utils or expected functions are missing.
    - It only updates fields when helpers are available.

    Expected (but not enforced) functions in `src.agent.extractors.time_utils`:
      - normalize_date(date_str: str) -> str
      - normalize_time(time_str: str) -> str
      - combine_to_iso(date_str: str, time_str: str) -> str
    """
    try:
        from src.agent.extractors import time_utils
    except Exception:
        return

    date_str = params.get("date")
    time_str = params.get("time")

    normalize_date = getattr(time_utils, "normalize_date", None)
    normalize_time = getattr(time_utils, "normalize_time", None)
    combine_to_iso = getattr(time_utils, "combine_to_iso", None)

    # Normalize date
    if callable(normalize_date) and isinstance(date_str, str):
        try:
            params["date"] = normalize_date(date_str)
        except Exception:
            pass

    # Normalize time
    if callable(normalize_time) and isinstance(time_str, str):
        try:
            params["time"] = normalize_time(time_str)
        except Exception:
            pass

    # Combine into an ISO datetime, if both parts are present and helper exists
    if (
        callable(combine_to_iso)
        and isinstance(params.get("date"), str)
        and isinstance(params.get("time"), str)
    ):
        try:
            params["datetime_iso"] = combine_to_iso(params["date"], params["time"])
        except Exception:
            pass


def extract_params(llm, user_input: str, intent: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract structured parameters from a user message using a generic LLM.

    Parameters
    ----------
    llm :
        A model instance created via `load_model(...)` or any compatible object.
        It must support either:
            - .invoke(prompt: str)
            - __call__(prompt: str)
    user_input : str
        Raw message written by the user in PT-BR.
    intent : Optional[str]
        Optional higher-level intent label (e.g., "check_availability",
        "create_reservation"). This can be used to give the model
        additional context about which parameters are most important, but the
        extractor remains generic if no intent is provided.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing at least:
            - "date": Optional[str]
            - "time": Optional[str]
            - "party_size": Optional[int]
            - "name": Optional[str]
            - "phone": Optional[str]
            - "email": Optional[str]
            - "notes": Optional[str]
            - "raw": Dict[str, Any]   (original parsed JSON from the model)
        It may also contain:
            - "datetime_iso": Optional[str] (if time_utils is available)
    """
    if intent:
        contextual_system_prompt = (
            SYSTEM_PROMPT
            + "\n\n"
            + f"Atenção: a intenção detectada para esta mensagem é: \"{intent}\".\n"
              "Priorize a extração de parâmetros relevantes para essa intenção."
        )
    else:
        contextual_system_prompt = SYSTEM_PROMPT

    prompt = build_single_turn_prompt(contextual_system_prompt, user_input)

    raw_text = call_llm(llm, prompt)
    parsed = _extract_json_from_text(raw_text)
    params = _ensure_param_schema(parsed)

    return params