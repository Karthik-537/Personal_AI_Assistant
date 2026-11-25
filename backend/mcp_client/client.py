"""MCP client: owns the connection to one MCP server.

Supports two transports:
  - connect_local_mcp_server():  spawn a local server subprocess over stdio
  - connect_remote_mcp_server(): connect to a remote server over streamable HTTP

Tool *discovery* lives here; schema conversion and execution live in
tool_manager.py / tool_executor.py.
"""

from __future__ import annotations

import logging
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import Tool

logger = logging.getLogger(__name__)


class MCPClient:
    """Connects to a single MCP server (local stdio or remote HTTP).

    Usage:
        client = MCPClient()
        await client.connect_remote_mcp_server("https://example.com/mcp")
        # or: await client.connect_local_mcp_server()
        tools = await client.discover_tools()
        ...
        await client.disconnect()
    """

    def __init__(self, server_params: StdioServerParameters | None = None):
        self._params = server_params
        self.session: ClientSession | None = None
        self._tools: list[Tool] | None = None
        self._exit_stack = AsyncExitStack()

    async def connect_local_mcp_server(self) -> None:
        """Spawn the server subprocess and perform the MCP handshake."""
        if self._params is None:
            raise RuntimeError(
                "connect_local_mcp_server requires StdioServerParameters"
            )
        logger.info(
            "Connecting to local MCP server: %s %s",
            self._params.command,
            " ".join(self._params.args),
        )
        read, write = await self._exit_stack.enter_async_context(
            stdio_client(self._params)
        )
        session = await self._exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await session.initialize()
        self.session = session
        self._tools = None

    async def connect_remote_mcp_server(self, url: str) -> None:
        """Connect to a remote MCP server over streamable HTTP."""
        logger.info("Connecting to remote MCP server: %s", url)
        read, write, _ = await self._exit_stack.enter_async_context(
            streamablehttp_client(url=url)
        )
        session = await self._exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await session.initialize()
        self.session = session
        self._tools = None

    async def disconnect(self) -> None:
        """Close the session and terminate the server subprocess/connection."""
        await self._exit_stack.aclose()
        self.session = None
        self._tools = None

    async def discover_tools(self) -> list[Tool]:
        """Fetch tools/list once and cache the result for this connection."""
        if self.session is None:
            raise RuntimeError("MCPClient is not connected")
        if self._tools is None:
            result = await self.session.list_tools()
            self._tools = list(result.tools)
            logger.info(
                "Discovered %d tool(s): %s",
                len(self._tools),
                [t.name for t in self._tools],
            )
        return self._tools
