# src/agent/actions/api_availability.py

from __future__ import annotations

from typing import Any, Dict, Optional, List

from src.services.getin_api import GetInApiClient, GetInApiError
from src.core.config import settings


_client: Optional[GetInApiClient] = None

DEFAULT_UNIT_ID: Optional[str] = getattr(settings, "getin_default_unit_id", None)


def _get_client() -> GetInApiClient:
    """Lazily instantiate a shared GetInApiClient."""
    global _client
    if _client is None:
        _client = GetInApiClient()
    return _client


def _extract_available_sectors(api_response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract a deduplicated list of available sectors from the GET IN schedules response.

    Supports two response shapes:

    1) By-unit endpoint (/schedules/units/:unit_id), docs example:
        {
          "success": true,
          "data": [
            {
              "hour": "12:00",
              "people": 2,
              "sector_id": "J1bqDWPO",
              "sector_name": "Salão 01",
              "discount": 0,
              "flexible": false
            },
            ...
          ]
        }

       In this case, each item in `data` is a schedule entry and we collect
       unique (sector_id, sector_name) pairs.

    2) Multi-unit endpoint (/schedules/units):
        {
          "success": true,
          "data": [
            {
              "id": "VPzzDDPQ",
              "name": "Get In AI 1",
              "schedules": [ ... ],
              "suggestions": [ ... ]
            },
            ...
          ]
        }

       In this case, we iterate units and then their `schedules`.

    Additionally, when `data` is empty but `suggestions` contains
    sector_id/sector_name, we also treat those sectors as "available"
    from suggestions (e.g. flexible seating).
    """
    sectors: List[Dict[str, Any]] = []
    seen = set()

    raw_data = api_response.get("data")

    if isinstance(raw_data, list) and raw_data:
        first = raw_data[0]

        if isinstance(first, dict) and (
            "sector_id" in first or "sectorId" in first
        ) and "hour" in first:
            for sch in raw_data:
                if not isinstance(sch, dict):
                    continue
                sector_id = sch.get("sector_id") or sch.get("sectorId")
                sector_name = sch.get("sector_name") or sch.get("sectorName")
                if not sector_id or not sector_name:
                    continue

                key = str(sector_id)
                if key in seen:
                    continue
                seen.add(key)

                sectors.append(
                    {
                        "source": "data",
                        "unit_id": None,
                        "unit_name": None,
                        "sector_id": str(sector_id),
                        "sector_name": str(sector_name),
                    }
                )

            if sectors:
                return sectors

        for unit in raw_data:
            if not isinstance(unit, dict):
                continue
            unit_id = unit.get("id")
            unit_name = unit.get("name")
            schedules = unit.get("schedules", [])
            if not isinstance(schedules, list):
                continue

            for sch in schedules:
                if not isinstance(sch, dict):
                    continue
                sector_id = sch.get("sector_id") or sch.get("sectorId")
                sector_name = sch.get("sector_name") or sch.get("sectorName")
                if not sector_id or not sector_name:
                    continue

                key = (str(unit_id), str(sector_id))
                if key in seen:
                    continue
                seen.add(key)

                sectors.append(
                    {
                        "source": "data",
                        "unit_id": str(unit_id) if unit_id is not None else None,
                        "unit_name": unit_name,
                        "sector_id": str(sector_id),
                        "sector_name": str(sector_name),
                    }
                )

        if sectors:
            return sectors

    raw_suggestions = api_response.get("suggestions")
    if isinstance(raw_suggestions, list):
        for item in raw_suggestions:
            if not isinstance(item, dict):
                continue
            sector_id = item.get("sector_id") or item.get("sectorId")
            sector_name = item.get("sector_name") or item.get("sectorName")
            if not sector_id or not sector_name:
                continue

            key = str(sector_id)
            if key in seen:
                continue
            seen.add(key)

            sectors.append(
                {
                    "source": "suggestions",
                    "unit_id": None,
                    "unit_name": None,
                    "sector_id": str(sector_id),
                    "sector_name": str(sector_name),
                }
            )

    return sectors


def check_availability(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check table availability via GET IN `/schedules/units/:unit_id` endpoint.

    This action:
      - resolves the unit_id (from params or default settings),
      - normalizes the time to HH:MM when possible,
      - calls the GET IN API,
      - extracts available sectors (unit_id + sector_id + sector_name).

    The returned payload is suitable for answer generation and for
    later steps in the reservation flow (POC 4).
    """
    unit_id = params.get("unit_id") or DEFAULT_UNIT_ID

    date = params.get("date")
    time = params.get("time")
    party_size = params.get("party_size")

    if isinstance(time, str):
        try:
            from src.agent.extractors.time_utils import normalize_time  # type: ignore
        except Exception:
            normalize_time = None

        if callable(normalize_time):
            try:
                time = normalize_time(time)
            except Exception:
                pass

    request_payload: Dict[str, Any] = {
        "unit_id": unit_id,
        "date": date,
        "time": time,
        "party_size": party_size,
    }

    # Basic validation
    missing: List[str] = []
    if not unit_id:
        missing.append("unit_id")
    if not date:
        missing.append("date")
    if not time:
        missing.append("time")
    if not party_size:
        missing.append("party_size")

    if missing:
        return {
            "type": "availability",
            "ok": False,
            "error": {
                "code": "MISSING_PARAMS",
                "message": (
                    "Parâmetros obrigatórios ausentes para checar disponibilidade: "
                    + ", ".join(missing)
                ),
                "details": {
                    "missing_fields": missing,
                },
            },
            "request": request_payload,
        }

    client = _get_client()

    try:
        api_response = client.list_schedules_units_by_unit(
            unit_id=str(unit_id),
            date=str(date),
            hour=str(time),
            people=int(party_size),
        )

        available_sectors = _extract_available_sectors(api_response)

        data_list = api_response.get("data") or []
        has_exact_slots = isinstance(data_list, list) and len(data_list) > 0

        return {
            "type": "availability",
            "ok": True,
            "request": request_payload,
            "response": api_response,
            "available_sectors": available_sectors,
            "has_exact_slots": has_exact_slots,
        }

    except GetInApiError as exc:
        return {
            "type": "availability",
            "ok": False,
            "error": {
                "code": "API_ERROR",
                "message": "Falha ao consultar disponibilidade na GET IN API.",
                "details": {
                    "status_code": exc.status_code,
                    "response_body": exc.response_body,
                    "exception_message": str(exc),
                },
            },
            "request": request_payload,
        }
    except Exception as exc:
        return {
            "type": "availability",
            "ok": False,
            "error": {
                "code": "UNEXPECTED_ERROR",
                "message": "Ocorreu um erro inesperado ao consultar disponibilidade.",
                "details": {
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                },
            },
            "request": request_payload,
        }