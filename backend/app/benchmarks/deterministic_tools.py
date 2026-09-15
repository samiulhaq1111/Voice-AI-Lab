"""Deterministic tool handlers for benchmark scenarios.

These handlers return fixed, predictable values so that benchmark
runs are repeatable. They are registered in a separate benchmark
tool registry and do NOT affect normal application behaviour.
"""

import asyncio
from typing import Any

from app.tools.tool import Tool

# ---------------------------------------------------------------------------
# Deterministic handlers
# ---------------------------------------------------------------------------

async def bench_get_weather_handler(location: str = "Unknown") -> dict[str, Any]:
    """Deterministic weather lookup — always returns the same values."""
    await asyncio.sleep(0.01)
    return {
        "location": location,
        "temperature_f": "72",
        "condition": "sunny",
    }


async def bench_get_employee_handler(employee_id: str = "") -> dict[str, str]:
    """Deterministic employee lookup — static table, no randomness."""
    await asyncio.sleep(0.01)
    employees = {
        "E001": {"name": "Alice Johnson", "department": "Engineering", "title": "Senior Developer"},
        "E002": {"name": "Bob Smith", "department": "Marketing", "title": "Marketing Manager"},
        "E003": {"name": "Carol Davis", "department": "HR", "title": "HR Specialist"},
    }
    default = {"name": "Unknown", "department": "Unknown", "title": "Unknown"}
    emp = employees.get(employee_id, default)
    return {"employee_id": employee_id, **emp}


async def bench_get_leave_balance_handler(
    employee_id: str = "",
    leave_type: str = "pto",
) -> dict[str, str]:
    """Deterministic leave balance — fixed values per employee/type."""
    await asyncio.sleep(0.01)
    balances = {
        ("E001", "pto"): "15",
        ("E001", "sick"): "7",
        ("E001", "vacation"): "10",
        ("E002", "pto"): "12",
        ("E002", "sick"): "5",
        ("E002", "vacation"): "8",
        ("E003", "pto"): "18",
        ("E003", "sick"): "6",
        ("E003", "vacation"): "12",
    }
    remaining = balances.get((employee_id, leave_type), "0")
    return {
        "employee_id": employee_id,
        "leave_type": leave_type,
        "remaining_days": remaining,
    }


# ---------------------------------------------------------------------------
# Registry builder
# ---------------------------------------------------------------------------

def create_benchmark_tools() -> list[Tool]:
    """Create deterministic tool handlers for benchmark scenarios."""
    return [
        Tool(
            name="get_weather",
            description="Get the current weather for a location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name or zip code",
                    },
                },
                "required": ["location"],
            },
            handler=bench_get_weather_handler,
        ),
        Tool(
            name="get_employee",
            description="Look up employee information by ID",
            parameters={
                "type": "object",
                "properties": {
                    "employee_id": {
                        "type": "string",
                        "description": "Employee ID (e.g. E001)",
                    },
                },
                "required": ["employee_id"],
            },
            handler=bench_get_employee_handler,
        ),
        Tool(
            name="get_leave_balance",
            description="Get remaining leave balance for an employee",
            parameters={
                "type": "object",
                "properties": {
                    "employee_id": {
                        "type": "string",
                        "description": "Employee ID (e.g. E001)",
                    },
                    "leave_type": {
                        "type": "string",
                        "description": "Type of leave: pto, sick, or vacation",
                        "enum": ["pto", "sick", "vacation"],
                    },
                },
                "required": ["employee_id", "leave_type"],
            },
            handler=bench_get_leave_balance_handler,
        ),
    ]
