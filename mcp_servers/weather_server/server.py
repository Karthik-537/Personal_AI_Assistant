"""Weather MCP server — exposes Open-Meteo functionality as MCP tools.

Run standalone for testing:
    python mcp_servers/weather_server/server.py

Transport: stdio (default for local MCP servers).
"""

from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP

from .weather_service import WeatherServiceError, get_forecast, search_locations

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("weather")


@mcp.tool()
async def search_location(location_name: str) -> list[dict]:
    """Search for a city or location by name.

    Use this FIRST when the user mentions a place by name. Returns matching
    locations with coordinates and timezone — pass the chosen match's
    latitude, longitude and timezone to get_weather_forecast.

    Args:
        location_name: City or place name, e.g. "Hyderabad" or "New York".
    """
    logger.info("search_location: %s", location_name)
    try:
        results = await search_locations(location_name)
    except WeatherServiceError as exc:
        logger.error("search_location failed: %s", exc)
        raise
    if not results:
        logger.info("search_location: no matches for %r", location_name)
    return results


@mcp.tool()
async def get_weather_forecast(
    latitude: float,
    longitude: float,
    timezone: str,
    forecast_days: int = 1,
) -> dict:
    """Get current weather and forecast for coordinates.

    Call search_location first to resolve a place name into coordinates.
    Returns current conditions, a daily forecast, and the next 24 hours of
    hourly data. Weather codes follow the WMO code table
    (0 = clear sky, 1-3 = partly/mostly cloudy, 45/48 = fog,
    51-67 = drizzle/rain, 71-77 = snow, 80-82 = showers, 95-99 = thunderstorm).

    Args:
        latitude: Latitude in decimal degrees (-90 to 90).
        longitude: Longitude in decimal degrees (-180 to 180).
        timezone: IANA timezone name, e.g. "Asia/Kolkata".
        forecast_days: Number of forecast days, 1-7 (default 1).
    """
    if not -90 <= latitude <= 90:
        raise ValueError(f"latitude out of range: {latitude}")
    if not -180 <= longitude <= 180:
        raise ValueError(f"longitude out of range: {longitude}")
    if not timezone.strip():
        raise ValueError("timezone must not be empty")
    forecast_days = max(1, min(7, forecast_days))

    logger.info(
        "get_weather_forecast: lat=%s lon=%s tz=%s days=%d",
        latitude, longitude, timezone, forecast_days,
    )
    try:
        return await get_forecast(latitude, longitude, timezone, forecast_days)
    except WeatherServiceError as exc:
        logger.error("get_weather_forecast failed: %s", exc)
        raise


if __name__ == "__main__":
    mcp.run()
