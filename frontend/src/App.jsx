import { useEffect, useMemo, useRef, useState } from "react";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from "chart.js";
import MapboxGlobe from "./components/MapboxGlobe";
import globalLocations from "./data/globalLocations";

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

const API_BASE = "http://127.0.0.1:8000";

// makeIcon removed — marker creation now handled inside MapboxGlobe.jsx

const REGION_COLORS = { arid: "#f59e0b", temperate: "#22d3ee", tropical: "#4ade80" };
const RISK_COLORS = { low: "#4ade80", medium: "#fbbf24", high: "#f87171" };

function getRainRegion(rain) {
  if (rain < 250) return "arid";
  if (rain <= 800) return "temperate";
  return "tropical";
}

// ─── API helpers ─────────────────────────────────────────────────────────────
function createApiError({ kind, path, message, status = null, details = null }) {
  const err = new Error(message);
  err.kind = kind; err.path = path; err.status = status; err.details = details;
  return err;
}

function classifyError(err, path) {
  if (err?.kind) return err;
  if (err?.name === "AbortError") return createApiError({ kind: "network", path, message: `Request to ${path} was aborted.` });
  if (err instanceof TypeError) return createApiError({ kind: "network", path, message: `Network error calling ${path}. Is the backend running?` });
  return createApiError({ kind: "unknown", path, message: err?.message || `Unexpected error calling ${path}.` });
}

function getErrorMessage(err) {
  if (!err) return "Unknown error";
  if (err.kind === "network") return err.message;
  if (err.kind === "backend") return `${err.path} failed (${err.status ?? "error"}): ${err.details || err.message}`;
  return err.message || "Unexpected error";
}

async function postJson(path, payload) {
  console.log("[API →]", path, payload ?? {});
  let res;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload ?? {})
    });
  } catch (err) {
    throw classifyError(err, path);
  }
  const text = await res.text();
  let parsed = null;
  try { parsed = text ? JSON.parse(text) : null; } catch { parsed = null; }

  if (!res.ok) {
    const detail = (parsed && (parsed.detail || parsed.message || JSON.stringify(parsed))) || text || "No response body";
    throw createApiError({ kind: "backend", path, status: res.status, details: detail, message: `Backend returned ${res.status}` });
  }
  console.log("[API ←]", path, parsed);
  return parsed ?? {};
}

// ─── Sub-components ───────────────────────────────────────────────────────────
function Spinner() {
  return (
    <div className="spinner-wrap">
      <div className="spinner" />
      <p>Fetching live analytics…</p>
    </div>
  );
}

function RegionBadge({ region }) {
  if (!region) return null;
  const color = REGION_COLORS[region] || "#94a3b8";
  return <span className="region-badge" style={{ background: color + "22", border: `1px solid ${color}88`, color }}>{region}</span>;
}

function RiskBadge({ risk }) {
  if (!risk || risk === "N/A") return <span className="risk-badge risk-unknown">Unknown</span>;
  const color = RISK_COLORS[risk.toLowerCase()] || "#94a3b8";
  return <span className="risk-badge" style={{ background: color + "22", border: `1px solid ${color}88`, color }}>{risk.toUpperCase()}</span>;
}

function TopCropsPanel({ topCrops, confidenceScores }) {
  if (!topCrops || topCrops.length === 0) return null;
  return (
    <div className="top-crops-panel">
      <h3>🌾 Top Crop Recommendations</h3>
      {topCrops.map((crop, i) => {
        const pct = Math.round((confidenceScores?.[i] ?? 0) * 100);
        const isTop = i === 0;
        return (
          <div key={crop} className={`crop-item ${isTop ? "crop-item--top" : ""}`}>
            <div className="crop-item-header">
              <span className="crop-rank">#{i + 1}</span>
              <span className="crop-name">{crop.charAt(0).toUpperCase() + crop.slice(1)}</span>
              <span className="crop-pct">{pct}%</span>
            </div>
            <div className="confidence-track">
              <div
                className="confidence-bar"
                style={{
                  width: `${pct}%`,
                  background: isTop
                    ? "linear-gradient(90deg, #22d3ee, #6366f1)"
                    : "linear-gradient(90deg, #4b5563, #6b7280)"
                }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [selectedCity, setSelectedCity] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [modelHealth, setModelHealth] = useState(null);
  const [metricsHistory, setMetricsHistory] = useState([]);
  const [metricsRefreshedAt, setMetricsRefreshedAt] = useState("");
  const requestSeqRef = useRef(0);

  const fetchModelHealth = async () => {
    try {
      const [metricsRes, historyRes] = await Promise.all([
        fetch(`${API_BASE}/metrics`),
        fetch(`${API_BASE}/metrics/history`)
      ]);
      if (!metricsRes.ok) throw new Error(`metrics failed: ${metricsRes.status}`);
      const data = await metricsRes.json();
      setModelHealth(data.artifacts ?? null);
      setMetricsRefreshedAt(data.generated_at ?? new Date().toISOString());
      if (historyRes.ok) {
        const h = await historyRes.json();
        setMetricsHistory(Array.isArray(h.history) ? h.history : []);
      } else {
        setMetricsHistory([]);
      }
    } catch {
      setModelHealth(null);
      setMetricsHistory([]);
      setMetricsRefreshedAt("");
    }
  };

  useEffect(() => { fetchModelHealth(); }, []);

  const getStatusLabel = (entry) => {
    if (!entry) return "Unknown";
    if (entry.available) return "Online";
    if (entry.error) return "Error";
    return "Missing";
  };

  const getStatusClass = (entry) => {
    if (!entry) return "status-unknown";
    if (entry.available) return "status-online";
    if (entry.error) return "status-error";
    return "status-missing";
  };

  const metricText = (metrics) => {
    if (!metrics) return "No metrics";
    if (typeof metrics.holdout_accuracy === "number")
      return `Acc ${metrics.holdout_accuracy.toFixed(2)} | F1 ${(metrics.holdout_f1_weighted ?? 0).toFixed(2)}`;
    if (typeof metrics.accuracy === "number")
      return `Acc ${metrics.accuracy.toFixed(2)} | F1 ${(metrics.f1_score ?? 0).toFixed(2)}`;
    return "Metrics tracked";
  };

  const metricValueForTrend = (metrics) => {
    if (!metrics) return { key: "", value: null, direction: "neutral" };
    if (typeof metrics.rmse === "number") return { key: "rmse", value: metrics.rmse, direction: "lower_better" };
    if (typeof metrics.holdout_rmse === "number") return { key: "holdout_rmse", value: metrics.holdout_rmse, direction: "lower_better" };
    if (typeof metrics.holdout_accuracy === "number") return { key: "holdout_accuracy", value: metrics.holdout_accuracy, direction: "higher_better" };
    if (typeof metrics.accuracy === "number") return { key: "accuracy", value: metrics.accuracy, direction: "higher_better" };
    return { key: "", value: null, direction: "neutral" };
  };

  const trendForModel = (modelKey, currentMetrics) => {
    const last = metricsHistory.at(-1)?.artifacts?.[modelKey]?.metrics;
    const prev = metricsHistory.at(-2)?.artifacts?.[modelKey]?.metrics;
    const current = metricValueForTrend(currentMetrics);
    const previous = metricValueForTrend(prev ?? last);
    if (!current.key || current.value == null || previous.value == null)
      return { symbol: "•", label: "No trend", className: "trend-flat" };
    const delta = Number(current.value) - Number(previous.value);
    if (Math.abs(delta) < 1e-9) return { symbol: "→", label: "Stable", className: "trend-flat" };
    if (current.direction === "lower_better")
      return delta < 0 ? { symbol: "↓", label: "Improving", className: "trend-up" } : { symbol: "↑", label: "Regressing", className: "trend-down" };
    return delta > 0 ? { symbol: "↑", label: "Improving", className: "trend-up" } : { symbol: "↓", label: "Regressing", className: "trend-down" };
  };

  const handleCityClick = async (city) => {
    const requestId = ++requestSeqRef.current;
    setSelectedCity(city);
    setLoading(true);
    setError("");
    setResult(null);

    // ── Validate inputs ──
    const temp = Number(city?.temp);
    const rain = Number(city?.rain);
    const hum = Number(city?.humidity ?? 60);

    if (!Number.isFinite(temp) || !Number.isFinite(rain) || !Number.isFinite(hum)) {
      setError("Invalid climate data for this location.");
      setLoading(false);
      return;
    }

    // ── FIX: classify-yield payload has ONLY 3 fields (no crop_type!) ──
    const classifyPayload = { temperature: temp, rainfall: rain, humidity: hum };

    const yieldPayload = { temperature: temp, rainfall: rain, humidity: hum, crop_type: "Wheat" };
    const clusterPayload = {
      samples: [{
        rainfall: rain,
        temperature: temp,
        N: Number(city?.N ?? 75),
        P: Number(city?.P ?? 40),
        K: Number(city?.K ?? 38)
      }]
    };

    try {
      const calls = await Promise.allSettled([
        postJson("/predict-yield", yieldPayload),
        postJson("/classify-yield", classifyPayload),   // ← fixed: no crop_type
        postJson("/forecast", { city: city.name, lat: city.lat, lng: city.lng }),
        postJson("/recommend", {
            temperature: temp,
            rainfall: rain,
            humidity: hum,
            n: Number(city?.N ?? 75),
            p: Number(city?.P ?? 40),
            k: Number(city?.K ?? 38)
        }),
        postJson("/cluster", clusterPayload)
      ]);

      const [yieldRes, classRes, forecastRes, recommendRes, clusterRes] = calls;

      const failures = calls
        .filter((r) => r.status === "rejected")
        .map((r) => classifyError(r.reason, "unknown"))
        .map((e) => getErrorMessage(e));

      const yieldResp = yieldRes.status === "fulfilled" ? yieldRes.value : null;
      const classResp = classRes.status === "fulfilled" ? classRes.value : null;
      const forecastResp = forecastRes.status === "fulfilled" ? forecastRes.value : null;
      const recommendResp = recommendRes.status === "fulfilled" ? recommendRes.value : null;
      const clusterResp = clusterRes.status === "fulfilled" ? clusterRes.value : null;

      const forecastPoints = Array.isArray(forecastResp?.forecast)
        ? forecastResp.forecast.map((p, i) => {
            if (typeof p === "number") return { date: `Step ${i + 1}`, rainfall: Number(p) || 0, prediction: Number(p) || 0 };
            return { date: p?.date ?? `Step ${i + 1}`, rainfall: Number(p?.rainfall ?? p?.prediction ?? 0), prediction: Number(p?.prediction ?? p?.rainfall ?? 0) };
          })
        : [];

      if (requestId !== requestSeqRef.current) return;

      setResult({
        cropSuitabilityScore: Number(yieldResp?.crop_suitability_score ?? yieldResp?.prediction ?? 0),
        riskLevel: classResp?.risk_level ?? "high",
        regionType: classResp?.region_type ?? getRainRegion(rain),
        topCrops: recommendResp?.realistic_recommendations ?? recommendResp?.recommendations ?? classResp?.top_crops ?? [],
        confidenceScores: classResp?.confidence_scores ?? [],
        projectedSuitability: forecastResp?.suitability_projection ?? null,
        rainfallHistory: Array.isArray(forecastResp?.historical) ? forecastResp.historical : [],
        forecast: forecastPoints,
        recommendation: recommendResp ?? { realistic_recommendations: [] },
        cluster: clusterResp ?? { clusters: [] }
      });

      if (failures.length > 0) {
        setError(`Partial data loaded. ${failures.join(" | ")}`);
      } else {
        setError("");
      }

      await fetchModelHealth();
    } catch (err) {
      if (requestId !== requestSeqRef.current) return;
      setError(getErrorMessage(classifyError(err, "dashboard")));
      setResult({
        cropSuitabilityScore: 0,
        riskLevel: "high",
        regionType: getRainRegion(city?.rain ?? 0),
        topCrops: [],
        confidenceScores: [],
        projectedSuitability: null,
        rainfallHistory: [],
        forecast: [],
        recommendation: { realistic_recommendations: [] },
        cluster: { clusters: [] }
      });
    } finally {
      if (requestId === requestSeqRef.current) setLoading(false);
    }
  };

  const chartData = useMemo(() => {
    const history = result?.rainfallHistory ?? [];
    const points = result?.forecast ?? [];
    const historyLabels = history.map((p) => p.date);
    const forecastLabels = points.map((p) => p.date);
    const labels = [...historyLabels, ...forecastLabels];
    const historyData = [...history.map((p) => p.rainfall), ...forecastLabels.map(() => null)];
    const forecastData = [...historyLabels.map(() => null), ...points.map((p) => p.rainfall ?? p.prediction)];
    return {
      labels,
      datasets: [
        { label: "Historical Rainfall", data: historyData, borderColor: "#94a3b8", backgroundColor: "rgba(148,163,184,0.15)", tension: 0.2, pointRadius: 1.5, fill: false },
        { label: "Forecast Rainfall", data: forecastData, borderColor: "#22d3ee", backgroundColor: "rgba(34,211,238,0.2)", tension: 0.3, pointRadius: 2, borderDash: [6, 3], fill: false }
      ]
    };
  }, [result]);

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { labels: { color: "#c7d2fe" } } },
    scales: {
      x: { ticks: { color: "#9ca3af" }, grid: { color: "rgba(255,255,255,0.08)" } },
      y: { ticks: { color: "#9ca3af" }, grid: { color: "rgba(255,255,255,0.08)" }, title: { display: true, text: "Rainfall (mm)", color: "#cbd5e1" } }
    }
  };

  // cityIcons memo removed — icon creation handled inside MapboxGlobe.jsx

  return (
    <div className="dashboard">
      <aside className="sidebar">
        {/* Header */}
        <div className="panel glass">
          <h1>🌍 Global Agri Intelligence</h1>
          <p className="muted">Region-aware ML crop predictions across 120+ global locations. Powered by RandomForest + climate analysis.</p>
          <div className="legend-row">
            <span className="legend-dot" style={{ background: REGION_COLORS.arid }} /> Arid (&lt;250mm)
            <span className="legend-dot" style={{ background: REGION_COLORS.temperate }} /> Temperate
            <span className="legend-dot" style={{ background: REGION_COLORS.tropical }} /> Tropical (&gt;800mm)
          </div>
        </div>

        {/* Results panel */}
        <div className={`panel glass transition ${loading ? "dimmed" : ""}`}>
          {!selectedCity && <p className="muted center-text">📍 Click any map marker to load live analytics.</p>}
          {loading && <Spinner />}
          {!loading && error && <div className="error-box">{error}</div>}

          {!loading && result && selectedCity && (
            <div className="metrics premium">
              <div className="city-header">
                <h2>{selectedCity.name}</h2>
                <div className="city-badges">
                  <RegionBadge region={result.regionType} />
                  <RiskBadge risk={result.riskLevel} />
                </div>
              </div>

              <div className="climate-stats">
                <div className="climate-stat">
                  <span>🌡️ Temp</span>
                  <strong>{selectedCity.temp}°C</strong>
                </div>
                <div className="climate-stat">
                  <span>🌧️ Rainfall</span>
                  <strong>{selectedCity.rain}mm</strong>
                </div>
                <div className="climate-stat">
                  <span>💧 Humidity</span>
                  <strong>{selectedCity.humidity ?? 60}%</strong>
                </div>
              </div>

              <div className="metric-grid">
                <div className="metric-card score-card">
                  <span>Yield Score</span>
                  <strong>{Number(result.cropSuitabilityScore ?? 0).toFixed(2)}</strong>
                </div>
                <div className="metric-card risk-card">
                  <span>Risk Level</span>
                  <strong style={{ color: RISK_COLORS[result.riskLevel] || "#f8fafc" }}>
                    {result.riskLevel?.toUpperCase() || "HIGH"}
                  </strong>
                </div>
                <div className="metric-card">
                  <span>Cluster</span>
                  <strong>{result?.cluster?.clusters?.[0] ?? "—"}</strong>
                </div>
              </div>

              {/* TOP 3 CROPS */}
              <TopCropsPanel topCrops={result.topCrops} confidenceScores={result.confidenceScores} />

              {result?.projectedSuitability && (
                <div className="projection-box">
                  <p><strong>Projected next-month score:</strong>{" "}{Number(result.projectedSuitability?.crop_suitability_score ?? 0).toFixed(2)}</p>
                  <p><strong>Projected risk:</strong> {result.projectedSuitability?.risk_level || "—"}</p>
                </div>
              )}

              {/* Model Health */}
              <div className="health-box">
                <div className="health-header">
                  <h3>Model Health</h3>
                  <button className="refresh-btn" onClick={fetchModelHealth} type="button">Refresh</button>
                </div>
                {metricsRefreshedAt && (
                  <p className="muted health-time">Updated: {new Date(metricsRefreshedAt).toLocaleString()}</p>
                )}
                {!modelHealth && <p className="muted">Model metrics unavailable.</p>}
                {modelHealth && (
                  <>
                    {Object.entries(modelHealth).map(([name, entry]) => (
                      <div className="health-row" key={name}>
                        <span>{name.replaceAll("_", " ")}</span>
                        <strong className={`status-badge ${getStatusClass(entry)}`}>{getStatusLabel(entry)}</strong>
                        <em>{metricText(entry?.metrics)}</em>
                        <small className={`trend-chip ${trendForModel(name, entry?.metrics).className}`}>
                          {trendForModel(name, entry?.metrics).symbol} {trendForModel(name, entry?.metrics).label}
                        </small>
                      </div>
                    ))}
                  </>
                )}
              </div>

              {/* Forecast Chart */}
              <div className="chart-wrap">
                <h3>Rainfall Forecast</h3>
                {(result?.forecast ?? []).length > 0 ? (
                  <div className="chart-container">
                    <Line data={chartData} options={chartOptions} />
                  </div>
                ) : (
                  <p className="muted">No forecast data for this location.</p>
                )}
              </div>
            </div>
          )}
        </div>
      </aside>

      {/* 3D Globe — Mapbox GL JS */}
      <main className="map-area">
        <MapboxGlobe
          cities={globalLocations}
          onCitySelect={handleCityClick}
        />
      </main>
    </div>
  );
}
