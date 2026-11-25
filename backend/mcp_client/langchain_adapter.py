import json

from typing import Any
from pydantic import Field, create_model

from langchain_core.tools import StructuredTool

from backend.mcp_client.tool_executor import execute_tool


def json_schema_to_type(schema: dict) -> type:
    """Convert basic JSON Schema types into Python types."""

    schema_type = schema.get("type", "string")

    type_mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "object": dict,
        "array": list,
    }

    return type_mapping.get(schema_type, Any)


def create_args_schema(tool):
    """Create a Pydantic schema from an MCP tool input schema."""

    schema = tool.inputSchema

    properties = schema.get("properties", {})
    required = schema.get("required", [])

    fields = {}

    for field_name, field_schema in properties.items():

        python_type = json_schema_to_type(field_schema)

        description = field_schema.get("description", "")

        if field_name in required:
            fields[field_name] = (
                python_type,
                Field(..., description=description)
            )
        else:
            fields[field_name] = (
                python_type,
                Field(
                    default=None,
                    description=description
                )
            )

    return create_model(
        f"{tool.name.title().replace('-', '')}Input",
        **fields
    )


def create_langchain_tool(mcp_tool, mcp_session):
    """Convert one MCP tool into an asynchronous LangChain tool."""

    tool_name = mcp_tool.name
    description = mcp_tool.description or tool_name

    args_schema = create_args_schema(mcp_tool)

    async def execute(**kwargs):

        result_json = await execute_tool(
            mcp_session,
            tool_name,
            kwargs
        )

        return result_json

    return StructuredTool(
        name=tool_name,
        description=description,
        args_schema=args_schema,
        coroutine=execute
    )


def convert_mcp_tools_to_langchain(mcp_tools, mcp_session):
    """Convert all discovered MCP tools into LangChain tools."""

    return [
        create_langchain_tool(tool, mcp_session)
        for tool in mcp_tools
    ]