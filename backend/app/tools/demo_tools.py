"""Demo tools for initial implementation.

These are simple mock tools for testing the tool-calling pipeline.
They do NOT integrate with real external services.
"""

import asyncio
import random

from app.tools.tool import Tool


async def get_weather_handler(location: str = "Unknown") -> dict[str, str]:
    """Mock weather tool handler."""
    await asyncio.sleep(0.1)  # Simulate API latency
    conditions = ["sunny", "cloudy", "rainy", "partly cloudy", "windy"]
    temp = random.randint(55, 95)
    return {
        "location": location,
        "temperature_f": str(temp),
        "condition": random.choice(conditions),
    }


async def get_employee_handler(employee_id: str = "") -> dict[str, str]:
    """Mock employee lookup tool handler."""
    await asyncio.sleep(0.05)
    employees = {
        "E001": {"name": "Alice Johnson", "department": "Engineering", "title": "Senior Developer"},
        "E002": {"name": "Bob Smith", "department": "Marketing", "title": "Marketing Manager"},
        "E003": {"name": "Carol Davis", "department": "HR", "title": "HR Specialist"},
    }
    default = {"name": "Unknown", "department": "Unknown", "title": "Unknown"}
    emp = employees.get(employee_id, default)
    return {"employee_id": employee_id, **emp}


async def get_leave_balance_handler(
    employee_id: str = "",
    leave_type: str = "pto",
) -> dict[str, str]:
    """Mock leave balance tool handler."""
    await asyncio.sleep(0.05)
    balances = {
        "pto": str(random.randint(5, 20)),
        "sick": str(random.randint(3, 10)),
        "vacation": str(random.randint(0, 15)),
    }
    return {
        "employee_id": employee_id,
        "leave_type": leave_type,
        "remaining_days": balances.get(leave_type, "0"),
    }


def create_demo_tools() -> list[Tool]:
    """Create and return all demo tools."""
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
            handler=get_weather_handler,
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
            handler=get_employee_handler,
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
            handler=get_leave_balance_handler,
        ),
    ]
