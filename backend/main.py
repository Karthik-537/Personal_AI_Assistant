"""Entry point: interactive chat with the Weather Agent.

Usage:
    python -m backend.main
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.agent.weather_agent import WeatherAgent
from backend.config import Settings
from backend.utils.logger import setup_logging


async def main() -> None:
    setup_logging()
    settings = Settings.from_env()
    agent = WeatherAgent(settings)
    await agent.connect()
    print("Weather Agent ready. Type a question, or 'quit' to exit.\n")
    try:
        while True:
            try:
                message = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not message:
                continue
            if message.lower() in {"quit", "exit"}:
                break
            try:
                answer = await agent.chat(message)
            except Exception as exc:  # noqa: BLE001 — keep REPL alive
                print(f"Agent error: {exc}\n")
                continue
            print(f"Agent: {answer}\n")
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
