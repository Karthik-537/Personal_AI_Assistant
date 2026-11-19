"""Application configuration — everything comes from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    gemini_model: str
    weather_server_command: str = "python"
    weather_server_args: tuple[str, ...] = ("-m", "mcp_servers.weather_server.server")
    max_tool_rounds: int = 10

    @classmethod
    def from_env(cls) -> "Settings":
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key or api_key == "your_api_key_here":
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and "
                "fill in your key."
            )
        return cls(
            gemini_api_key=api_key,
            gemini_model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
            max_tool_rounds=int(os.environ.get("MAX_TOOL_ROUNDS", "10")),
        )
