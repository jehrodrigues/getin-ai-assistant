# src/agent/actions/api_reservations.py

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, List

from src.core.config import settings
from src.services.getin_api import GetInApiClient, GetInApiError

DEFAULT_UNIT_ID: Optional[str] = settings.getin_default_unit_id

_client: Optional[GetInApiClient] = None


def _get_client() -> GetInApiClient:
    """
    Lazily instantiate a shared GetInApiClient.

    This avoids re-creating the HTTP client on every call.
    """
    global _client
    if _client is None:
        _client = GetInApiClient()
    return _client


def _resolve_sector_id_from_params(params: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """
    Try to resolve sector_id using:
      - explicit params["sector_id"], or
      - a single available_sectors entry, or
      - matching notes against sector_name.

    Returns:
        (sector_id, info)
    """
    sector_id = params.get("sector_id")
    info = params.get("notes")

    available_sectors: List[Dict[str, Any]] = params.get("available_sectors") or []

    if sector_id:
        return str(sector_id), str(info) if info is not None else None

    if len(available_sectors) == 1:
        sector = available_sectors[0]
        resolved_id = sector.get("sector_id")
        resolved_name = sector.get("sector_name")

        if resolved_id:
            if not info and resolved_name:
                info = resolved_name
            return str(resolved_id), str(info) if info is not None else None

    if info and available_sectors:
        normalized_info = str(info).strip().lower()
        for sector in available_sectors:
            name = str(sector.get("sector_name") or "").strip().lower()
            sid = sector.get("sector_id")
            if not name or not sid:
                continue

            if name in normalized_info or normalized_info in name:
                return str(sid), str(info)

    return None, str(info) if info is not None else None


def _maybe_normalize_time(raw_time: Any) -> Any:
    """
    Normalize time to HH:MM if time_utils is available.
    """
    if not isinstance(raw_time, str):
        return raw_time

    try:
        from src.agent.extractors.time_utils import normalize_time
    except Exception:
        normalize_time = None

    if callable(normalize_time):
        try:
            return normalize_time(raw_time)
        except Exception:
            return raw_time

    return raw_time


def create_reservation(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a new reservation via GET IN POST /reservations.

    Expected params (as produced by `extract_params` + availability context):
        - date: Optional[str]        (ISO "YYYY-MM-DD")
        - time: Optional[str]        ("HH:MM" or "20h" (will be normalized if possible))
        - party_size: Optional[int]
        - name: Optional[str]
        - phone: Optional[str]
        - email: Optional[str]
        - unit_id: Optional[str]     (optional; defaults to DEFAULT_UNIT_ID)
        - sector_id: Optional[str]   (can be auto-resolved from available_sectors + notes)
        - table_people: Optional[int] (defaults to party_size)
        - notes: Optional[str]       (mapped to "info")
        - available_sectors: Optional[List[Dict[str, Any]]] (injected by workflow)
    """
    client = _get_client()

    unit_id = params.get("unit_id") or DEFAULT_UNIT_ID
    date = params.get("date")
    time = _maybe_normalize_time(params.get("time"))
    party_size = params.get("party_size")
    name = params.get("name")
    phone = params.get("phone")
    email = params.get("email")

    resolved_sector_id, resolved_info = _resolve_sector_id_from_params(params)

    sector_id = resolved_sector_id
    table_people = params.get("table_people") or party_size
    info = resolved_info
    discount = params.get("discount")
    custom_fields = params.get("custom_fields")
    highlighted = params.get("highlighted")

    request_payload: Dict[str, Any] = {
        "date": date,
        "time": time,
        "party_size": party_size,
        "name": name,
        "phone": phone,
        "email": email,
        "unit_id": unit_id,
        "sector_id": sector_id,
        "table_people": table_people,
        "info": info,
        "discount": discount,
        "custom_fields": custom_fields,
        "highlighted": highlighted,
    }

    missing = []
    if not unit_id:
        missing.append("unit_id")
    if not sector_id:
        missing.append("sector_id")
    if not date:
        missing.append("date")
    if not time:
        missing.append("time")
    if not party_size:
        missing.append("party_size")
    if not name:
        missing.append("name")
    if not phone:
        missing.append("phone")
    if not email:
        missing.append("email")

    if missing:
        return {
            "type": "create_reservation",
            "ok": False,
            "error": {
                "code": "MISSING_PARAMS",
                "message": (
                    "Faltam algumas informações para criar a reserva: "
                    + ", ".join(missing)
                ),
                "details": {"missing_fields": missing},
            },
            "request": request_payload,
        }

    try:
        people_int = int(party_size)
    except Exception:
        return {
            "type": "create_reservation",
            "ok": False,
            "error": {
                "code": "INVALID_PARAM",
                "message": "O número de pessoas informado é inválido.",
                "details": {"party_size": party_size},
            },
            "request": request_payload,
        }

    try:
        table_people_int = int(table_people)
    except Exception:
        return {
            "type": "create_reservation",
            "ok": False,
            "error": {
                "code": "INVALID_PARAM",
                "message": "A capacidade de mesa (table_people) é inválida.",
                "details": {"table_people": table_people},
            },
            "request": request_payload,
        }

    discount_val: Optional[float] = None
    if discount is not None:
        try:
            discount_val = float(discount)
        except Exception:
            discount_val = None

    custom_fields_val: Optional[List[Dict[str, Any]]] = None
    if isinstance(custom_fields, list):
        custom_fields_val = custom_fields

    highlighted_val: Optional[bool] = None
    if isinstance(highlighted, bool):
        highlighted_val = highlighted

    try:
        api_response = client.create_reservation(
            unit_id=str(unit_id),
            sector_id=str(sector_id),
            name=str(name),
            mobile=str(phone),
            email=str(email),
            people=people_int,
            table_people=table_people_int,
            date=str(date),
            time=str(time),
            info=str(info) if info is not None else None,
            discount=discount_val,
            custom_fields=custom_fields_val,
            highlighted=highlighted_val,
        )
        return {
            "type": "create_reservation",
            "ok": True,
            "request": request_payload,
            "response": api_response,
        }

    except GetInApiError as exc:
        return {
            "type": "create_reservation",
            "ok": False,
            "error": {
                "code": "API_ERROR",
                "message": "Falha ao criar a reserva na GET IN API.",
                "details": {
                    "status_code": exc.status_code,
                    "response_body": exc.response_body,
                    "exception_message": str(exc),
                },
            },
            "request": request_payload,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "type": "create_reservation",
            "ok": False,
            "error": {
                "code": "UNEXPECTED_ERROR",
                "message": "Ocorreu um erro inesperado ao criar a reserva.",
                "details": {
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                },
            },
            "request": request_payload,
        }


def view_next_reservation(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retrieve the next reservation(s) for a given user via GET IN /reservations/next.

    For POC 3/4, we primarily filter by phone or e-mail.
    """
    client = _get_client()

    unit_id = params.get("unit_id") or DEFAULT_UNIT_ID
    phone = params.get("phone")
    email = params.get("email")
    start_date = params.get("start_date")
    end_date = params.get("end_date")
    status = params.get("status")

    request_payload: Dict[str, Any] = {
        "phone": phone,
        "email": email,
        "unit_id": unit_id,
        "start_date": start_date,
        "end_date": end_date,
        "status": status,
    }

    if not phone and not email:
        return {
            "type": "view_next_reservation",
            "ok": False,
            "error": {
                "code": "MISSING_PARAMS",
                "message": (
                    "Para localizar sua próxima reserva, preciso de um telefone "
                    "ou e-mail associado às reservas."
                ),
                "details": {"required_one_of": ["phone", "email"]},
            },
            "request": request_payload,
        }

    try:
        api_response = client.get_next_reservations(
            unit_id=unit_id,
            mobile=phone,
            email=email,
            start_date=start_date,
            end_date=end_date,
            status=status,
            sort=None,
            page=None,
        )
        return {
            "type": "view_next_reservation",
            "ok": True,
            "request": request_payload,
            "response": api_response,
        }

    except GetInApiError as exc:
        return {
            "type": "view_next_reservation",
            "ok": False,
            "error": {
                "code": "API_ERROR",
                "message": "Falha ao consultar a próxima reserva na GET IN API.",
                "details": {
                    "status_code": exc.status_code,
                    "response_body": exc.response_body,
                    "exception_message": str(exc),
                },
            },
            "request": request_payload,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "type": "view_next_reservation",
            "ok": False,
            "error": {
                "code": "UNEXPECTED_ERROR",
                "message": "Ocorreu um erro inesperado ao consultar sua próxima reserva.",
                "details": {
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                },
            },
            "request": request_payload,
        }


def list_reservations(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Consult reservations using GET IN /reservations.
    """
    client = _get_client()

    unit_id = params.get("unit_id") or DEFAULT_UNIT_ID
    search_val = params.get("search")
    phone = params.get("phone")
    email = params.get("email")
    date = params.get("date")
    start_date = params.get("start_date")
    end_date = params.get("end_date")
    status = params.get("status")
    sector_id = params.get("sector_id")
    sort = params.get("sort")
    page = params.get("page")

    request_payload: Dict[str, Any] = {
        "unit_id": unit_id,
        "search": search_val,
        "phone": phone,
        "email": email,
        "date": date,
        "start_date": start_date,
        "end_date": end_date,
        "status": status,
        "sector_id": sector_id,
        "sort": sort,
        "page": page,
    }

    if not any([phone, email, search_val, date, start_date, end_date]):
        return {
            "type": "list_reservations",
            "ok": False,
            "error": {
                "code": "MISSING_FILTERS",
                "message": (
                    "Para consultar reservas, preciso de pelo menos uma informação "
                    "como telefone, e-mail, nome ou data."
                ),
                "details": {
                    "required_one_of": ["phone", "email", "search", "date", "start_date", "end_date"],
                },
            },
            "request": request_payload,
        }

    try:
        result = client.list_reservations(
            unit_id=unit_id,
            search=search_val,
            mobile=phone,
            email=email,
            date=date,
            start_date=start_date,
            end_date=end_date,
            status=status,
            sector_id=sector_id,
            sort=sort,
            page=page,
        )
        return {
            "type": "list_reservations",
            "ok": True,
            "request": request_payload,
            "response": result.raw,
        }

    except GetInApiError as exc:
        return {
            "type": "list_reservations",
            "ok": False,
            "error": {
                "code": "API_ERROR",
                "message": "Falha ao consultar reservas na GET IN API.",
                "details": {
                    "status_code": exc.status_code,
                    "response_body": exc.response_body,
                    "exception_message": str(exc),
                },
            },
            "request": request_payload,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "type": "list_reservations",
            "ok": False,
            "error": {
                "code": "UNEXPECTED_ERROR",
                "message": "Ocorreu um erro inesperado ao consultar reservas.",
                "details": {
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                },
            },
            "request": request_payload,
        }


def cancel_reservation(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cancel an existing reservation using Option A:
    list -> disambiguate -> delete.
    """
    raise NotImplementedError("Keep your existing cancel_reservation implementation here.")