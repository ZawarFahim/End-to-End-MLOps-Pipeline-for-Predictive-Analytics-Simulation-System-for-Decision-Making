import requests
import logging

LOG = logging.getLogger(__name__)

def get_real_world_weather(lat: float, lon: float) -> dict:
    """Fetches real-world historical average weather from NASA POWER API."""
    try:
        url = "https://power.larc.nasa.gov/api/temporal/climatology/point"
        params = {
            "parameters": "T2M,PRECTOTCORR,RH2M", # Temp, Rain, Humidity
            "community": "AG",
            "longitude": lon,
            "latitude": lat,
            "format": "JSON"
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # NASA returns months and an annual average ("ANN"). We extract the annual baseline.
        parameters = data.get("properties", {}).get("parameter", {})
        
        # Extract Annual (ANN) averages. If missing, provide safe fallbacks.
        temp = parameters.get("T2M", {}).get("ANN", 25.0)
        rain = parameters.get("PRECTOTCORR", {}).get("ANN", 2.0) * 365 # Convert daily average to annual
        humidity = parameters.get("RH2M", {}).get("ANN", 60.0)
        
        return {
            "temperature": round(temp, 2),
            "rainfall": round(rain, 2),
            "humidity": round(humidity, 2)
        }
    except Exception as e:
        LOG.error(f"NASA API failed: {e}. Using fallback data.")
        # Fallback to temperate climate if NASA API goes down
        return {"temperature": 25.0, "rainfall": 500.0, "humidity": 60.0}
