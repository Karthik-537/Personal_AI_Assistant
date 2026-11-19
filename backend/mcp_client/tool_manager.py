"""Convert MCP tool definitions into Gemini function declarations.

This is the schema impedance-matching layer: MCP speaks JSON Schema,
Gemini's google-genai SDK wants types.Schema / types.FunctionDeclaration.
Nothing here is hardcoded per-tool — it works for ANY MCP server.
"""

from __future__ import annotations

import logging

from google.genai import types
from mcp.types import Tool

logger = logging.getLogger(__name__)

# JSON Schema type -> Gemini Schema type
_TYPE_MAP = {
    "string": types.Type.STRING,
    "number": types.Type.NUMBER,
    "integer": types.Type.INTEGER,
    "boolean": types.Type.BOOLEAN,
    "object": types.Type.OBJECT,
    "array": types.Type.ARRAY,
    "null": types.Type.STRING,  # Gemini has no null type; treat as string
}


def _convert_schema(schema: dict) -> types.Schema:
    """Recursively convert a JSON Schema fragment to a Gemini types.Schema."""
    json_type = schema.get("type", "object")
    gemini_type = _TYPE_MAP.get(json_type, types.Type.STRING)

    kwargs: dict = {
        "type": gemini_type,
        "description": schema.get("description"),
        "nullable": json_type == "null" or schema.get("nullable", False),
    }

    if "enum" in schema:
        kwargs["enum"] = [str(v) for v in schema["enum"]]

    if json_type == "object" or "properties" in schema:
        properties = schema.get("properties", {})
        kwargs["properties"] = {
            name: _convert_schema(prop) for name, prop in properties.items()
        }
        required = schema.get("required", [])
        kwargs["required"] = list(required)

    if json_type == "array" and "items" in schema:
        kwargs["items"] = _convert_schema(schema["items"])

    return types.Schema(**{k: v for k, v in kwargs.items() if v is not None})


def tool_to_function_declaration(tool: Tool) -> types.FunctionDeclaration:
    """Convert one MCP Tool into a Gemini FunctionDeclaration."""
    return types.FunctionDeclaration(
        name=tool.name,
        description=tool.description or "",
        parameters=_convert_schema(tool.inputSchema or {"type": "object"}),
    )


def tools_to_function_declarations(tools: list[Tool]) -> list[types.FunctionDeclaration]:
    """Convert a list of MCP tools; logs what was produced."""
    declarations = [tool_to_function_declaration(t) for t in tools]
    logger.info(
        "Converted %d MCP tool(s) to Gemini declarations: %s",
        len(declarations),
        [d.name for d in declarations],
    )
    return declarations
