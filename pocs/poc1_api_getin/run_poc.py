import sys

from src.services.getin_api import GetInApiClient, GetInApiError


def run_poc(search: str | None = "Get In") -> int:
    """
    POC 1: Validate connectivity and authentication using the GetInApiClient
    against the /units endpoint.

    Returns:
        0 if the POC succeeds
        1 if it fails
    """
    client = GetInApiClient()

    print("Running POC 1 – GET IN API connectivity test (/units via GetInApiClient)")
    print(f"Base URL: {client.base_url}")
    print(f"Search term: {search!r}")

    try:
        result = client.list_units(search=search)
    except GetInApiError as exc:
        print("POC 1 failed due to API error.")
        print(f"Status code: {exc.status_code}")
        print(f"Message: {exc}")
        print(f"Response body: {exc.response_body}")
        return 1

    print("Raw JSON response:", result.raw)

    print(f"Units returned: {len(result.units)}")
    for unit in result.units:
        print(f"- {unit.id} – {unit.name} – {unit.full_address}")

    print("Pagination:", result.pagination)

    print("POC 1 completed successfully.")
    return 0


def main() -> None:
    """
    Entry point for running POC 1.

    Usage:
        python -m pocs.poc1_api_getin.run_poc
        python -m pocs.poc1_api_getin.run_poc "Get In AI"
    """
    search: str | None = "Get In"
    if len(sys.argv) >= 2:
        search = sys.argv[1]

    exit_code = run_poc(search=search)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()