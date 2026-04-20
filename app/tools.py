from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from langchain_core.tools import tool


@tool("current_utc_datetime")
def current_utc_datetime() -> str:
    """Return the current UTC date and time as an ISO 8601 string."""

    return datetime.now(timezone.utc).isoformat()


DEFAULT_TOOL_REGISTRY: dict[str, Any] = {
    current_utc_datetime.name: current_utc_datetime,
}
