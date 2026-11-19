"""Integration test: spawn the weather MCP server over stdio and exercise it.

This validates the full MCP path without Gemini:
    test process -> stdio_client -> server subprocess -> Open-Meteo

Requires network access to Open-Meteo (free, no key).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _server_params() -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_servers.weather_server.server"],
        cwd=str(PROJECT_ROOT),
    )


@pytest.mark.asyncio
async def test_tools_list_and_call():
    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            # 1. Initialize handshake (MCP initialize -> notifications/initialized)
            await session.initialize()

            # 2. tools/list — dynamic discovery
            tools = await session.list_tools()
            names = {t.name for t in tools.tools}
            assert names == {"search_location", "get_weather_forecast"}

            # 3. tools/call — geocoding
            result = await session.call_tool(
                "search_location", {"location_name": "Hyderabad"}
            )
            assert not result.isError
            assert result.content
            text = result.content[0].text
            assert "Hyderabad" in text
            assert "latitude" in text

            # 4. tools/call — forecast, using coordinates from step 3
            import json

            matches = json.loads(text)
            first = matches[0]
            forecast = await session.call_tool(
                "get_weather_forecast",
                {
                    "latitude": first["latitude"],
                    "longitude": first["longitude"],
                    "timezone": first["timezone"],
                    "forecast_days": 1,
                },
            )
            assert not forecast.isError
            fc_text = forecast.content[0].text
            assert "current" in fc_text
            assert "temperature_2m" in fc_text
