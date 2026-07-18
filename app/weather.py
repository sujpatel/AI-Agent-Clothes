import requests


def get_weather(location: str) -> dict:
    geo_resp = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": location, "count": 1},
        timeout=10,
    ).json()

    results = geo_resp.get("results")
    if not results:
        return {"error": f"Could not find location: {location}"}

    lat = results[0]["latitude"]
    lon = results[0]["longitude"]

    weather_resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,precipitation",
            "temperature_unit": "fahrenheit",
        },
        timeout=10,
    ).json()

    current = weather_resp["current"]
    return {
        "location": location,
        "temperature_f": current["temperature_2m"],
        "precipitation_mm": current["precipitation"],
    }
