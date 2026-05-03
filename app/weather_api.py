import requests
import logging
from datetime import date, timedelta

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


def get_nasa_daily_timeseries(
    lat: float,
    lon: float,
    *,
    days: int = 60,
) -> dict:
    """
    Fetch daily temporal series from NASA POWER (historical daily).

    Returns:
      {
        "historical": [{"date": "YYYY-MM-DD", "rainfall": mm, "temperature": C, "humidity": %}, ...],
        "source": "nasa_power_daily"
      }
    """
    days = int(days)
    if days <= 0:
        raise ValueError("days must be > 0")
    if days > 3650:
        days = 3650

    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=days - 1)

    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": "T2M,PRECTOTCORR,RH2M",
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
        "format": "JSON",
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    p = data.get("properties", {}).get("parameter", {}) or {}

    t2m = p.get("T2M", {}) or {}
    pr = p.get("PRECTOTCORR", {}) or {}
    rh = p.get("RH2M", {}) or {}

    keys = sorted(set(t2m.keys()) | set(pr.keys()) | set(rh.keys()))
    historical = []
    for k in keys:
        # keys are typically YYYYMMDD
        try:
            d = f"{k[0:4]}-{k[4:6]}-{k[6:8]}"
        except Exception:
            continue
        rain = pr.get(k, None)
        temp = t2m.get(k, None)
        hum = rh.get(k, None)
        if rain is None and temp is None and hum is None:
            continue
        historical.append(
            {
                "date": d,
                "rainfall": float(rain) if rain is not None else None,
                "temperature": float(temp) if temp is not None else None,
                "humidity": float(hum) if hum is not None else None,
            }
        )

    return {"historical": historical, "source": "nasa_power_daily", "start": str(start), "end": str(end)}
