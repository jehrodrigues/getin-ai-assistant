from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from src.core.config import settings


class GetInApiError(Exception):
    """Custom exception for GET IN API errors."""

    def __init__(self, message: str, status_code: int, response_body: Any) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


@dataclass
class Unit:
    """Domain representation of a GET IN unit."""
    id: str
    name: str
    city_slug: Optional[str] = None
    slug: Optional[str] = None
    full_address: Optional[str] = None
    timezone: Optional[str] = None
    telephone: Optional[str] = None

    @classmethod
    def from_api(cls, raw: Dict[str, Any]) -> "Unit":
        """Create a Unit instance from the raw API payload."""
        return cls(
            id=str(raw.get("id")),
            name=raw.get("name", ""),
            city_slug=raw.get("city_slug"),
            slug=raw.get("slug"),
            full_address=raw.get("full_address"),
            timezone=raw.get("timezone"),
            telephone=raw.get("telephone"),
        )


@dataclass
class ListUnitsResult:
    """Structured result for the /units endpoint."""
    units: List[Unit]
    pagination: Dict[str, Any]
    raw: Dict[str, Any]


@dataclass
class ListReservationsResult:
    """Structured result for the /reservations endpoint."""
    reservations: List[Dict[str, Any]]
    pagination: Dict[str, Any]
    raw: Dict[str, Any]


class GetInApiClient:
    """
    HTTP client wrapper for GET IN v2 APIs.

    Responsibilities:
    - Handle base URL and authentication headers
    - Provide typed methods for key endpoints (e.g. /units, /schedules/units, /reservations)
    - Centralise error handling and response validation
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 10,
    ) -> None:
        self.api_key = api_key or settings.getin_api_key
        self.base_url = (base_url or settings.getin_api_base_url).rstrip("/")
        self.timeout = timeout

        if not self.api_key:
            raise ValueError("GET IN api_key is not configured.")
        if not self.base_url:
            raise ValueError("GET IN base_url is not configured.")

    @property
    def _headers(self) -> Dict[str, str]:
        """Default headers for all GET IN API requests."""
        return {
            "apiKey": self.api_key,
            "Accept": "application/json",
        }

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Internal helper to perform an HTTP request and return JSON.

        Raises:
            GetInApiError if the status code is not 2xx or the body is not JSON.
        """
        url = f"{self.base_url}/{path.lstrip('/')}"
        try:
            response = requests.request(
                method=method.upper(),
                url=url,
                headers=self._headers,
                params=params,
                json=json,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise GetInApiError(
                message=f"Network error calling GET IN API: {exc}",
                status_code=0,
                response_body=str(exc),
            ) from exc

        try:
            data = response.json()
        except ValueError:
            raise GetInApiError(
                message=f"Invalid JSON response from GET IN API. Status: {response.status_code}",
                status_code=response.status_code,
                response_body=response.text,
            )

        if not response.ok:
            msg = data.get("message") if isinstance(data, dict) else None
            message = msg or f"GET IN API request failed with status {response.status_code}"
            raise GetInApiError(
                message=message,
                status_code=response.status_code,
                response_body=data,
            )

        return data

    def list_units(
        self,
        search: Optional[str] = None,
        coordinates: Optional[str] = None,
        distance: Optional[int] = None,
        no_show_enabled: Optional[int] = None,
    ) -> ListUnitsResult:
        """
        Call the /units endpoint and return a structured result.

        Args:
            search: Filter by unit name.
            coordinates: Filter by latitude and longitude (format defined by API).
            distance: Distance radius (defaults to 10 in the API if omitted).
            no_show_enabled: Filter by no-show availability (1 enabled, 0 disabled).

        Returns:
            ListUnitsResult with parsed Unit objects and pagination metadata.
        """
        params: Dict[str, Any] = {}
        if search:
            params["search"] = search
        if coordinates:
            params["coordinates"] = coordinates
        if distance is not None:
            params["distance"] = distance
        if no_show_enabled is not None:
            params["no_show_enabled"] = no_show_enabled

        raw = self._request("GET", "/units", params=params)

        data_list = raw.get("data", [])
        pagination = raw.get("pagination", {})

        units = [Unit.from_api(item) for item in data_list]

        return ListUnitsResult(units=units, pagination=pagination, raw=raw)

    def list_schedules_units(
        self,
        date: str,
        hour: str,
        people: int,
    ) -> Dict[str, Any]:
        """
        Call the /schedules/units endpoint to check availability.

        Args:
            date: Date in YYYY-MM-DD format.
            hour: Time in HH:MM format.
            people: Number of people for availability search.

        Returns:
            Raw JSON dictionary with availability information.
        """
        params: Dict[str, Any] = {
            "date": date,
            "hour": hour,
            "people": people,
        }
        return self._request("GET", "/schedules/units", params=params)

    def list_schedules_units_by_unit(
            self,
            unit_id: str,
            date: str,
            hour: str,
            people: int,
    ) -> Dict[str, Any]:
        """
        Call the /schedules/units/:unit_id endpoint to check availability
        for a specific unit.

        Args:
            unit_id: Unit identifier (path parameter).
            date: Date in YYYY-MM-DD format.
            hour: Time in HH:MM format.
            people: Number of people for availability search.

        Returns:
            Raw JSON dictionary with availability information for that unit.
        """
        params: Dict[str, Any] = {
            "date": date,
            "hour": hour,
            "people": people,
        }
        path = f"/schedules/units/{unit_id}"
        return self._request("GET", path, params=params)

    def create_reservation(
        self,
        unit_id: str,
        sector_id: str,
        name: str,
        mobile: str,
        email: str,
        people: int,
        table_people: int,
        date: str,
        time: str,
        info: str | None = None,
        discount: float | None = None,
        custom_fields: list[dict[str, Any]] | None = None,
        highlighted: bool | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Call the POST /reservations endpoint to create a new reservation.

        The payload must follow the JSON structure expected by the GET IN API:

            - unit_id: ID of the unit. Obtained from Schedules / Retrieve Availability
                       or Schedules / Retrieve Availability By Unit (field: id).
            - sector_id: ID of the sector/room. Obtained from Schedules / Retrieve Availability
                         or Schedules / Retrieve Availability By Unit
                         (field: schedules[x].sector_id).
            - name: Name of the person making the reservation.
            - mobile: Phone number of the person making the reservation.
            - email: E-mail of the person making the reservation.
            - people: Number of people for the reservation.
            - table_people: Capacity of the table to be occupied. Obtained from
                            Schedules / Retrieve Availability or Schedules / Retrieve
                            Availability By Unit (field: schedules[x].people).
            - date: Date of the reservation (format defined by the API, usually YYYY-MM-DD).
            - time: Time of the reservation (format defined by the API, usually HH:MM).
            - info: Optional note/observation about the reservation. This will be shown
                    to the operator in the panel.
            - discount: Optional discount percentage, if configured in Get In Admin.
                        Obtained from Schedules / Retrieve Availability or
                        Schedules / Retrieve Availability By Unit
                        (field: schedules[x].discount).
            - custom_fields: Optional list of custom fields and their values, if configured
                             in Get In Admin. Obtained from
                             Reservations / Custom Fields / Retrieve All.
            - highlighted: Optional boolean indicating whether the reservation should be
                           highlighted in the panel.

        Any additional fields can be passed via the `extra` dict and will be merged into
        the JSON payload.

        Returns
        -------
        Dict[str, Any]
            Raw JSON response from the API.

        Raises
        ------
        GetInApiError
            If the request fails or the response is not a valid JSON.
        """
        payload: Dict[str, Any] = {
            "unit_id": unit_id,
            "sector_id": sector_id,
            "name": name,
            "mobile": mobile,
            "email": email,
            "people": people,
            "table_people": table_people,
            "date": date,
            "time": time,
        }

        if info is not None:
            payload["info"] = info
        if discount is not None:
            payload["discount"] = discount
        if custom_fields is not None:
            payload["custom_fields"] = custom_fields
        if highlighted is not None:
            payload["highlighted"] = highlighted

        if extra:
            payload.update(extra)

        return self._request("POST", "/reservations", json=payload)

    def delete_reservation(
        self,
        reservation_id: str,
        unit_id: str,
        sector_id: str,
        name: str,
        mobile: str,
        email: str,
        people: int,
        table_people: int,
        date: str,
        time: str,
        info: Optional[str] = None,
        discount: Optional[float] = None,
        custom_fields: Optional[List[Dict[str, Any]]] = None,
        highlighted: Optional[bool] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Call the DELETE /reservations/:reservation_id endpoint to delete an existing reservation.

        The payload must follow the same JSON structure expected by the GET IN API as for
        reservation creation:

            - unit_id: ID of the unit. Obtained from Schedules / Retrieve Availability
                       or Schedules / Retrieve Availability By Unit (field: id).
            - sector_id: ID of the sector/room. Obtained from Schedules / Retrieve Availability
                         or Schedules / Retrieve Availability By Unit
                         (field: schedules[x].sector_id).
            - name: Name of the person who made the reservation.
            - mobile: Phone number of the person who made the reservation.
            - email: E-mail of the person who made the reservation.
            - people: Number of people for the reservation.
            - table_people: Capacity of the table that was occupied. Obtained from
                            Schedules / Retrieve Availability or Schedules / Retrieve
                            Availability By Unit (field: schedules[x].people).
            - date: Date of the reservation.
            - time: Time of the reservation.
            - info: Optional note/observation about the reservation.
            - discount: Optional discount percentage, if configured in Get In Admin.
            - custom_fields: Optional list of custom fields and their values.
            - highlighted: Optional boolean indicating whether the reservation
                           should be highlighted in the panel.

        Any additional fields can be passed via the `extra` dict and will be merged into
        the JSON payload.

        Returns:
            Raw JSON response from the API.

        Raises:
            GetInApiError if the request fails or the response is not valid JSON.
        """
        payload: Dict[str, Any] = {
            "unit_id": unit_id,
            "sector_id": sector_id,
            "name": name,
            "mobile": mobile,
            "email": email,
            "people": people,
            "table_people": table_people,
            "date": date,
            "time": time,
        }

        if info is not None:
            payload["info"] = info
        if discount is not None:
            payload["discount"] = discount
        if custom_fields is not None:
            payload["custom_fields"] = custom_fields
        if highlighted is not None:
            payload["highlighted"] = highlighted

        if extra:
            payload.update(extra)

        path = f"/reservations/{reservation_id}"
        return self._request("DELETE", path, json=payload)

    def get_next_reservations(
        self,
        unit_id: Optional[str] = None,
        mobile: Optional[str] = None,
        email: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        status: Optional[str] = None,
        sort: Optional[str] = None,
        page: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Call the /reservations/next endpoint to retrieve the next reservation(s)
        for a given set of filters.

        Args:
            unit_id: Specific unit to filter by (text, length 8).
            mobile: Specific phone number to filter by.
            email: Specific e-mail address to filter by.
            start_date: Start of the date range to search (YYYY-MM-DD).
            end_date: End of the date range to search (YYYY-MM-DD).
            status: Reservation status to filter by. Possible values:
                    PENDING|CONFIRMED|CANCELED-USER|CANCELED_AGENT.
            sort: Sort expression, e.g. "field:direction" or
                  "field1:direction,field2:direction".
            page: Page number for paginated responses.

        Returns:
            Raw JSON dictionary with the next reservation(s) information.

        Raises:
            GetInApiError if the request fails or the response is not valid JSON.
        """
        params: Dict[str, Any] = {}

        if unit_id:
            params["unit_id"] = unit_id
        if mobile:
            params["mobile"] = mobile
        if email:
            params["email"] = email
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if status:
            params["status"] = status
        if sort:
            params["sort"] = sort
        if page is not None:
            params["page"] = page

        return self._request("GET", "/reservations/next", params=params)

    def list_reservations(
        self,
        unit_id: Optional[str] = None,
        search: Optional[str] = None,
        mobile: Optional[str] = None,
        email: Optional[str] = None,
        date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        status: Optional[str] = None,
        sector_id: Optional[str] = None,
        sort: Optional[str] = None,
        page: Optional[int] = None,
    ) -> ListReservationsResult:
        """
        Call the /reservations endpoint and return a structured result.

        Args:
            unit_id: Unit identifier (text, length 8).
            search: Generic contact field (name, e-mail, phone).
            mobile: Specific phone number.
            email: Specific e-mail address.
            date: Single date to query (YYYY-MM-DD).
            start_date: Start of date range (YYYY-MM-DD).
            end_date: End of date range (YYYY-MM-DD).
            status: Reservation status (PENDING|CONFIRMED|CANCELED-USER|CANCELED_AGENT).
            sector_id: Specific room/sector identifier (text, length 8).
            sort: Sort expression, e.g. "field:direction" or "field1:direction,field2:direction".
            page: Page number for paginated responses.

        Returns:
            ListReservationsResult with the reservation list and pagination metadata.
        """
        params: Dict[str, Any] = {}

        if unit_id:
            params["unit_id"] = unit_id
        if search:
            params["search"] = search
        if mobile:
            params["mobile"] = mobile
        if email:
            params["email"] = email
        if date:
            params["date"] = date
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if status:
            params["status"] = status
        if sector_id:
            params["sector_id"] = sector_id
        if sort:
            params["sort"] = sort
        if page is not None:
            params["page"] = page

        raw = self._request("GET", "/reservations", params=params)

        reservations = raw.get("data", [])
        pagination = raw.get("pagination", {})

        if not isinstance(reservations, list):
            raise GetInApiError(
                message="Unexpected /reservations response format: 'data' is not a list.",
                status_code=200,
                response_body=raw,
            )

        return ListReservationsResult(
            reservations=reservations,
            pagination=pagination,
            raw=raw,
        )