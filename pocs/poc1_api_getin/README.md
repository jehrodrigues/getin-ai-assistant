# PoC 1 – GET IN API Connectivity Validation

**Goal**

Validate that the application can successfully connect to the **GET IN v2 APIs**
and retrieve structured data using authenticated requests.

Specifically, this PoC validates that the system can:

- load configuration from environment variables
- authenticate against GET IN APIs using an API key
- establish a working HTTP client for GET IN v2
- query the `/units` endpoint
- parse and display structured unit data returned by the API
- operate independently from LLMs, RAG, and agent workflows

This PoC establishes the **external integration baseline** required for all
subsequent PoCs that depend on GET IN data.

**Example interactions**

```text
Running PoC 1 without filters:

Found 2 units:
- Get In AI 1 (VPzzDDPQ)
- Get In AI 2 (D1Z5zOPm)

Running PoC 1 with unit name filter:

Input: "Get In AI 1"
Output:
- ID: VPzzDDPQ
  Name: Get In AI 1
  Address: Avenida Valentim Gentil, 470, São Paulo-SP