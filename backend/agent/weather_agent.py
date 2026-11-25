import logging
import sys

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from langgraph.graph import (
    StateGraph,
    START,
    END,
    MessagesState
)

from langgraph.prebuilt import ToolNode, tools_condition

from mcp import StdioServerParameters

from backend.config import Settings
from backend.mcp_client.client import MCPClient

from backend.mcp_client.langchain_adapter import (
    convert_mcp_tools_to_langchain
)


logger = logging.getLogger(__name__)


class WeatherAgent:

    def __init__(self, settings: Settings):

        self._settings = settings

        # 1. Initialize Gemini through LangChain
        self._llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            temperature=0.2
        )

        # 2. Initialize existing MCP client
        self._mcp = MCPClient(
            StdioServerParameters(
                command=sys.executable,
                args=list(settings.weather_server_args)
            )
        )

        # 3. Initialize agent state
        self._history = []

        # 4. Graph and tools will be initialized after MCP connection
        self._tools = []
        self._llm_with_tools = None
        self._graph = None

    async def connect(self) -> None:
        """Connect to MCP and build the LangGraph workflow."""

        # Connect to weather MCP server (remote URL if configured, else local stdio)
        if self._settings.weather_mcp_url:
            await self._mcp.connect_remote_mcp_server(
                url=self._settings.weather_mcp_url
            )
        else:
            await self._mcp.connect_local_mcp_server()

        # Discover MCP tools
        mcp_tools = await self._mcp.discover_tools()

        logger.info(
            "Discovered %s MCP tools",
            len(mcp_tools)
        )

        # Convert MCP tools into LangChain tools
        self._tools = convert_mcp_tools_to_langchain(
            mcp_tools,
            self._mcp.session
        )

        # Bind tools to Gemini
        self._llm_with_tools = self._llm.bind_tools(
            self._tools
        )

        # Build LangGraph
        self._build_graph()

    def _build_graph(self) -> None:
        """Construct and compile the LangGraph workflow."""

        # Assistant node
        def assistant(state: MessagesState):

            system_message = SystemMessage(
                content="""
                You are an intelligent weather assistant.

                Use the available weather tools to answer
                user questions about weather.

                Instructions:
                1. Call appropriate MCP tools when necessary.
                2. Never invent weather information.
                3. Use tool results to generate your answer.
                4. If a tool fails, explain the failure.
                5. Provide clear and concise answers.
                """
            )

            response = self._llm_with_tools.invoke(
                [system_message] + state["messages"]
            )

            return {
                "messages": [response]
            }

        # Create graph
        builder = StateGraph(MessagesState)

        # Register nodes
        builder.add_node("assistant", assistant)

        builder.add_node(
            "tools",
            ToolNode(
                self._tools,
                handle_tool_errors=True
            )
        )

        # Define edges
        builder.add_edge(START, "assistant")

        builder.add_conditional_edges(
            "assistant",
            tools_condition,
            {
                "tools": "tools",
                END: END
            }
        )

        builder.add_edge("tools", "assistant")

        # Compile graph
        self._graph = builder.compile()

    async def chat(self, message: str) -> str:
        """Process a user message through LangGraph."""

        if self._graph is None:
            raise RuntimeError(
                "Weather agent is not connected. Call connect() first."
            )

        # Add user message to conversation
        self._history.append(
            HumanMessage(content=message)
        )

        # Execute graph
        result = await self._graph.ainvoke(
            {
                "messages": self._history
            },
            config={
                "recursion_limit": (
                    self._settings.max_tool_rounds * 2 + 5
                )
            }
        )

        # Update conversation history
        self._history = result["messages"]

        # Return final response — Gemini content may be a str or a list of
        # content blocks; normalize to plain text.
        final_message = self._history[-1]
        content = final_message.content
        if isinstance(content, list):
            return "".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        return content

    async def close(self) -> None:
        """Disconnect MCP server."""

        await self._mcp.disconnect()
