"""Weather agent: Gemini function-calling loop over MCP tools.

The agent owns two things:
  1. The Gemini conversation (contents history, multi-turn).
  2. The MCP connection to the weather server (tool discovery + execution).

Loop per user message:
    prompt + tool declarations -> Gemini
    Gemini returns function_call(s) -> MCP tools/call -> function_response
    -> Gemini ... until it returns plain text (the answer).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from google import genai
from google.genai import types
from mcp import StdioServerParameters

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import Settings
from backend.mcp_client.client import MCPClient
from backend.mcp_client.tool_executor import execute_tool
from backend.mcp_client.tool_manager import tools_to_function_declarations

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 10


class WeatherAgent:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._mcp = MCPClient(
            StdioServerParameters(
                command=sys.executable,
                args=list(settings.weather_server_args),
                cwd=str(PROJECT_ROOT),
            )
        )
        self._declarations: list[types.FunctionDeclaration] = []
        self._history: list[types.Content] = []

    async def connect(self) -> None:
        """Start the MCP server, discover tools, build Gemini declarations."""
        await self._mcp.connect()
        tools = await self._mcp.discover_tools()
        self._declarations = tools_to_function_declarations(tools)

    async def close(self) -> None:
        await self._mcp.disconnect()

    async def chat(self, message: str) -> str:
        """Send a user message; returns Gemini's final text answer."""
        self._history.append(
            types.Content(role="user", parts=[types.Part.from_text(text=message)])
        )

        config = types.GenerateContentConfig(
            tools=[types.Tool(function_declarations=self._declarations)]
        )

        for round_num in range(self._settings.max_tool_rounds):
            response = await self._client.aio.models.generate_content(
                model=self._settings.gemini_model,
                contents=self._history,
                config=config,
            )

            if not response.candidates:
                raise RuntimeError(
                    f"Gemini returned no candidates "
                    f"(finish_reason={getattr(response, 'prompt_feedback', None)})"
                )

            candidate = response.candidates[0]
            model_content = candidate.content
            if model_content is None:
                raise RuntimeError("Gemini returned no content")

            # model_content already contains any text AND function_call parts —
            # appending it once records the full model turn in history.
            self._history.append(model_content)

            function_calls = [
                p.function_call for p in (model_content.parts or []) if p.function_call
            ]
            if not function_calls:
                # Extract text safely — response.text raises if absent.
                texts = [
                    p.text
                    for p in (model_content.parts or [])
                    if getattr(p, "text", None)
                ]
                return "".join(texts)

            # Execute every requested tool call, then feed results back.
            response_parts = []
            for fc in function_calls:
                logger.info("Gemini requests tool: %s(%s)", fc.name, fc.args)
                try:
                    result_json = await execute_tool(
                        self._mcp.session, fc.name, dict(fc.args or {})
                    )
                except Exception as exc:  # noqa: BLE001 — report to Gemini, keep loop alive
                    logger.error("Tool %s failed: %s", fc.name, exc)
                    result_json = json.dumps({"error": str(exc)})
                response_parts.append(
                    types.Part.from_function_response(
                        name=fc.name,
                        response={"result": json.loads(result_json)},
                    )
                )

            self._history.append(
                types.Content(role="user", parts=response_parts)
            )
        else:
            raise RuntimeError(
                f"Exceeded {self._settings.max_tool_rounds} tool-call rounds"
            )
