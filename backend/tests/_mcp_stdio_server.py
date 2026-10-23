"""Minimal local stdio MCP server used only by MCP integration tests.

Spawned as a subprocess by the MCP lifecycle and agent integration tests
through the SDK's stdio transport. Exposes read-only, deterministic tools
(``echo`` and ``lookup_employee``) so tests can verify the bridge and the
agent tool-call loop against the real SDK wire shapes without any network
access. ``lookup_employee`` serves a small fictional dataset (placeholder
identities only — no real personal data) and performs no writes or
external calls.
"""

import json

from mcp.server import MCPServer

mcp = MCPServer("voice-lab-test")

# Fictional, deterministic records — placeholder identities only.
_EMPLOYEES = {
    "E100": {
        "id": "E100",
        "name": "Jane Example",
        "department": "Engineering",
        "location": "Remote",
    },
    "E200": {
        "id": "E200",
        "name": "John Sample",
        "department": "Marketing",
        "location": "Berlin",
    },
}


@mcp.tool()
def echo(text: str) -> str:
    """Echo the provided text back."""
    return f"echo: {text}"


@mcp.tool()
def lookup_employee(employee_id: str) -> str:
    """Look up a fictional employee by ID and return a JSON record."""
    record = _EMPLOYEES.get(employee_id.strip().upper())
    if record is None:
        return json.dumps({"error": f"unknown employee id: {employee_id}"})
    return json.dumps(record, sort_keys=True)


if __name__ == "__main__":
    mcp.run()
