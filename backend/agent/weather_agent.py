import logging
import sys
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain.mcp import MCPAdapter

from fastmcp.client.transports import StdioTransport

from backend.config import Settings


logger = logging.getLogger(__name__)
project_root = Path(__file__).resolve().parents[2]


class WeatherAgent:

    def __init__(self, settings: Settings):
        self._settings = settings
        self._llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            temperature=0.2
        )

    async def create_mcp_agent(self, server):
        adapter = MCPAdapter(server)
        await adapter.__aenter__()

        tools = await adapter.list_tools()

        agent = create_agent(
            model=self._llm,
            tools=tools
        )
        return agent, adapter

    async def chat(self, message: str) -> str:
        if self._settings.weather_mcp_url:
            agent, adapter = await self.create_mcp_agent(
                server=self._settings.weather_mcp_url)
        else:
            agent, adapter = await self.create_mcp_agent(
                server=StdioTransport(
                    command=sys.executable,
                    args=list(self._settings.weather_server_args),
                    cwd=str(project_root)
                )
            )

        try:
            response = await agent.ainvoke({
                "messages": [HumanMessage(content=message)]
            })
            content = response["messages"][-1].content

            if isinstance(content, list):
                content = "".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )

            return content
        finally:
            await adapter.__aexit__(None, None, None)
