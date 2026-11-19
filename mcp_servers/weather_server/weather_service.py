"""Async client for the Open-Meteo weather API (no API key required).

This module contains pure domain logic — it knows nothing about MCP.
The MCP server (server.py) adapts these functions into tools.
"""

from __future__ import annotations

import httpx

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Variables requested from the forecast API.
CURRENT_VARIABLES = (
    "temperature_2m,relative_humidity_2m,apparent_temperature,"
    "is_day,precipitation,weather_code,wind_speed_10m"
)
DAILY_VARIABLES = (
    "weather_code,temperature_2m_max,temperature_2m_min,"
    "apparent_temperature_max,apparent_temperature_min,"
    "precipitation_probability_max,precipitation_sum,wind_speed_10m_max"
)
HOURLY_VARIABLES = (
    "temperature_2m,apparent_temperature,precipitation_probability,"
    "precipitation,weather_code,wind_speed_10m"
)

DEFAULT_TIMEOUT = httpx.Timeout(15.0)


class WeatherServiceError(Exception):
    """Raised when an Open-Meteo request fails or returns unusable data."""


async def search_locations(location_name: str, count: int = 5) -> list[dict]:
    """Search for locations by name via the Open-Meteo geocoding API.

    Returns a list of matches, each with name, country, admin1 (state/region),
    latitude, longitude and timezone. Empty list means no matches.
    """
    params = {
        "name": location_name,
        "count": count,
        "language": "en",
        "format": "json",
    }
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await client.get(GEOCODING_URL, params=params)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise WeatherServiceError(f"Geocoding request failed: {exc}") from exc

    data = response.json()
    results = data.get("results") or []
    return [
        {
            "name": r.get("name"),
            "country": r.get("country"),
            "admin1": r.get("admin1"),
            "latitude": r.get("latitude"),
            "longitude": r.get("longitude"),
            "timezone": r.get("timezone"),
        }
        for r in results
    ]


async def get_forecast(
    latitude: float,
    longitude: float,
    timezone: str,
    forecast_days: int = 1,
) -> dict:
    """Fetch current conditions, daily forecast and next-24h hourly data.

    Weather codes follow the WMO code table (e.g. 0 = clear, 61 = rain).
    Hourly data is trimmed to the next 24 hours to keep payloads small.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "timezone": timezone,
        "forecast_days": forecast_days,
        "current": CURRENT_VARIABLES,
        "hourly": HOURLY_VARIABLES,
        "daily": DAILY_VARIABLES,
    }
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await client.get(FORECAST_URL, params=params)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise WeatherServiceError(f"Forecast request failed: {exc}") from exc

    data = response.json()
    if "error" in data or "reason" in data:
        raise WeatherServiceError(f"Forecast API error: {data}")

    # Trim hourly arrays to the next 24 entries (indices are aligned arrays).
    hourly = data.get("hourly") or {}
    trimmed_hourly = {
        key: values[:24] for key, values in hourly.items() if isinstance(values, list)
    }

    return {
        "location": {
            "latitude": data.get("latitude"),
            "longitude": data.get("longitude"),
            "timezone": data.get("timezone"),
            "timezone_abbreviation": data.get("timezone_abbreviation"),
            "elevation": data.get("elevation"),
        },
        "current": data.get("current"),
        "hourly": trimmed_hourly,
        "daily": data.get("daily"),
    }
