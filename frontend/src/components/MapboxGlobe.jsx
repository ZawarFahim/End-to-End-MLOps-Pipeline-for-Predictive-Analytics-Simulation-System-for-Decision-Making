import { useEffect, useRef } from "react";
import mapboxgl from "mapbox-gl";
import "mapbox-gl/dist/mapbox-gl.css";

const TOKEN =
  "pk.eyJ1IjoiemF3YXJmYWhpbSIsImEiOiJjbW9vdDlpdmExaTByMm9zMWNjaHFiZGx1In0.LAJunS_5s8lapn1RTiiVPA";

// ─── Build GeoJSON — integer `id` required for feature queries ───────────────
function buildGeoJSON(cities) {
  return {
    type: "FeatureCollection",
    features: cities.map((city, i) => ({
      type: "Feature",
      id: i,
      properties: {
        name:     city.name,
        temp:     city.temp,
        rain:     city.rain,
        humidity: city.humidity ?? 60,
        N:        city.N  ?? 75,
        P:        city.P  ?? 40,
        K:        city.K  ?? 38,
        lat:      city.lat,
        lng:      city.lng,
      },
      geometry: {
        type: "Point",
        coordinates: [city.lng, city.lat],
      },
    })),
  };
}

// ─── Popup content ────────────────────────────────────────────────────────────
function popupHtml(p) {
  return `<div style="font-family:Inter,sans-serif;font-size:13px;min-width:150px">
    <b style="font-size:15px">📍 ${p.name}</b><br/>
    <span style="color:#475569;font-size:12px">
      🌡️ ${p.temp}°C &nbsp;|&nbsp; 🌧️ ${p.rain} mm<br/>
      💧 ${p.humidity}% humidity
    </span>
    <div style="margin-top:5px;font-size:11px;color:#94a3b8;font-style:italic">
      Running ML analysis…
    </div>
  </div>`;
}

// ─── Component ────────────────────────────────────────────────────────────────
export default function MapboxGlobe({ cities, onCitySelect, flyToCoord, highlightedCity }) {
  const containerRef = useRef(null);
  const mapRef = useRef(null);
  const rotatingRef = useRef(true);
  const rotateTimeoutRef = useRef(null);
  const animationRef = useRef(null);
  const pauseRotationRef = useRef(false);
  const pulseRef = useRef({ t: 0, dir: 1 });
  // Keep onCitySelect stable — never a stale closure
  const onSelectRef  = useRef(onCitySelect);
  useEffect(() => { onSelectRef.current = onCitySelect; }, [onCitySelect]);

  useEffect(() => {
    if (!containerRef.current) return;

    mapboxgl.accessToken = TOKEN;

    const map = new mapboxgl.Map({
      container:  containerRef.current,
      style:      "mapbox://styles/mapbox/satellite-streets-v12",
      projection: "globe",
      zoom:       1.5,
      center:     [10, 20],
      antialias:  true,
    });
    mapRef.current = map;

    map.addControl(new mapboxgl.NavigationControl(), "top-right");

    // One shared popup
    const popup = new mapboxgl.Popup({
      closeButton:  true,
      closeOnClick: false,
      maxWidth:     "240px",
      className:    "globe-popup",
    });

    map.on("load", () => {
      // Starfield atmosphere
      map.setFog({
        color:            "rgb(186,210,235)",
        "high-color":     "rgb(36,92,223)",
        "horizon-blend":  0.02,
        "space-color":    "rgb(11,11,25)",
        "star-intensity": 0.6,
      });

      // ── GeoJSON source ──────────────────────────────────────────────────
      map.addSource("cities", {
        type: "geojson",
        data: buildGeoJSON(cities),
      });

      // ── Invisible larger hit-area layer (makes clicking easy) ───────────
      //    Rendered first (behind), transparent, 2× the visual radius
      map.addLayer({
        id:     "city-hit",
        type:   "circle",
        source: "cities",
        paint: {
          "circle-radius":            22,
          "circle-opacity":           0,       // fully invisible
          "circle-stroke-width":      0,
          "circle-pitch-alignment":   "viewport",
          "circle-pitch-scale":       "viewport",
        },
      });

      // ── Visible solid red dot ────────────────────────────────────────────
      map.addLayer({
        id:     "city-dots",
        type:   "circle",
        source: "cities",
        paint: {
          "circle-radius":            10,      // fixed 10 px — always clear
          "circle-color":             "#e52222",
          "circle-stroke-width":      2.5,
          "circle-stroke-color":      "#ffffff",
          "circle-opacity":           1,
          "circle-pitch-alignment":   "viewport",   // does NOT shrink on globe tilt
          "circle-pitch-scale":       "viewport",
        },
      });

      // ── Cursor feedback ──────────────────────────────────────────────────
      map.on("mouseenter", "city-hit", () => {
        map.getCanvas().style.cursor = "pointer";
        rotatingRef.current = false;
      });
      map.on("mouseleave", "city-hit", () => {
        map.getCanvas().style.cursor = "";
        rotatingRef.current = true;
      });

      // ── CLICK — fires on the hit layer (larger, easier to click) ────────
      map.on("click", "city-hit", (e) => {
        if (!e.features || e.features.length === 0) return;

        const p     = e.features[0].properties;
        const coord = e.features[0].geometry.coordinates.slice();

        // Reconstruct full city object (GeoJSON stores everything as strings)
        const city = {
          name:     p.name,
          lat:      Number(p.lat),
          lng:      Number(p.lng),
          temp:     Number(p.temp),
          rain:     Number(p.rain),
          humidity: Number(p.humidity),
          N:        Number(p.N),
          P:        Number(p.P),
          K:        Number(p.K),
        };

        // Fly to location
        map.flyTo({ center: coord, zoom: Math.max(map.getZoom(), 4), duration: 800 });

        // Popup pinned to exact coordinate — never drifts
        popup.setLngLat(coord).setHTML(popupHtml(p)).addTo(map);

        // ← Trigger existing ML pipeline in App.jsx
        onSelectRef.current(city);
      });

      const resumeRotation = () => {
        if (rotateTimeoutRef.current) clearTimeout(rotateTimeoutRef.current);
        rotateTimeoutRef.current = setTimeout(() => {
          rotatingRef.current = true;
          pauseRotationRef.current = false;
        }, 1500);
      };
      map.on("mousedown", () => {
        pauseRotationRef.current = true;
        rotatingRef.current = false;
      });
      map.on("mouseup", resumeRotation);
      map.on("dragstart", () => {
        pauseRotationRef.current = true;
        rotatingRef.current = false;
      });
      map.on("dragend", resumeRotation);

      const spin = () => {
        if (!mapRef.current) return;
        if (rotatingRef.current && !pauseRotationRef.current) {
          mapRef.current.rotateTo(
            mapRef.current.getBearing() + 0.05,
            { duration: 0 }
          );
        }

        // Pulse highlighted marker by animating circle-radius
        if (mapRef.current.getLayer("city-dots")) {
          let { t, dir } = pulseRef.current;
          t += dir * 0.04;
          if (t >= 1) { t = 1; dir = -1; }
          if (t <= 0) { t = 0; dir = 1; }
          pulseRef.current = { t, dir };
          const pulseRadius = 10 + (6 * t);
          mapRef.current.setPaintProperty("city-dots", "circle-radius", [
            "case",
            ["==", ["downcase", ["to-string", ["get", "name"]]], String(highlightedCity || "").toLowerCase()],
            pulseRadius,
            10,
          ]);
          mapRef.current.setPaintProperty("city-dots", "circle-color", [
            "case",
            ["==", ["downcase", ["to-string", ["get", "name"]]], String(highlightedCity || "").toLowerCase()],
            "#ff0000",
            "#e52222",
          ]);
        }
        animationRef.current = requestAnimationFrame(spin);
      };
      animationRef.current = requestAnimationFrame(spin);
    });

    // Cleanup
    return () => {
      popup.remove();
      if (rotateTimeoutRef.current) clearTimeout(rotateTimeoutRef.current);
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
      map.remove();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!mapRef.current || !flyToCoord) return;
    mapRef.current.flyTo({ center: [flyToCoord.lng, flyToCoord.lat], zoom: 4, duration: 1200 });
  }, [flyToCoord]);

  useEffect(() => {
    if (!mapRef.current || !mapRef.current.getLayer("city-dots")) return;
    mapRef.current.setPaintProperty("city-dots", "circle-color", [
      "case",
      ["==", ["downcase", ["to-string", ["get", "name"]]], String(highlightedCity || "").toLowerCase()],
      "#ff0000",
      "#e52222",
    ]);
  }, [highlightedCity]);

  return (
    <div
      ref={containerRef}
      style={{ width: "100%", height: "100%", borderRadius: "inherit" }}
    />
  );
}
