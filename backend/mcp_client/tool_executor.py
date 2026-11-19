"""Execute MCP tool calls and normalize results for Gemini.

Gemini expects tool responses as strings. MCP returns CallToolResult whose
content blocks hold text (usually a JSON string for our tools). This module
unwraps that into a clean JSON string, and surfaces MCP-level errors.
"""

from __future__ import annotations

import json
import logging

from mcp import ClientSession
from mcp.shared.exceptions import McpError

logger = logging.getLogger(__name__)


async def execute_tool(
    session: ClientSession,
    name: str,
    arguments: dict,
) -> str:
    """Call an MCP tool and return its result as a JSON string.

    Raises McpError if the server reports a tool-level error; raises
    RuntimeError for transport/protocol failures.
    """
    logger.info("Executing MCP tool %r with args %s", name, arguments)
    result = await session.call_tool(name, arguments)

    if result.isError:
        # Extract the error text from content blocks if present.
        error_text = ""
        if result.content:
            error_text = getattr(result.content[0], "text", "") or ""
        raise McpError(f"Tool '{name}' returned an error: {error_text}")

    # Preferred path: structuredContent holds the COMPLETE result. FastMCP
    # serializes unstructured list returns as one TextContent per item, so
    # reading only content[0] would silently drop data.
    if result.structuredContent is not None:
        return json.dumps(result.structuredContent, ensure_ascii=False)

    if not result.content:
        return "{}"

    # Fallback: concatenate all text blocks (non-JSON or legacy servers).
    texts = [
        c.text for c in result.content if getattr(c, "text", None) is not None
    ]
    combined = "\n".join(texts)
    try:
        return json.dumps(json.loads(combined), ensure_ascii=False)
    except json.JSONDecodeError:
        return json.dumps({"result": combined}, ensure_ascii=False)
