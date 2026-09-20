import { useState, useRef } from "react";
import HeroBackground from "./components/HeroBackground";

import {
  MapContainer,
  TileLayer,
  Rectangle,
  useMapEvents,
  useMap,
} from "react-leaflet";

import "leaflet/dist/leaflet.css";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

const DEFAULT_CENTER = [22.72, 75.85];

function MapSelector({ onSelect }) {
  const [start, setStart] = useState(null);
  const [current, setCurrent] = useState(null);

  const drawing = useRef(false);

  const map = useMap();

  useMapEvents({
    mousedown(e) {
      if (e.originalEvent.button !== 0) {
        return;
      }

      // Disable map panning while the user draws the box,
      // otherwise Leaflet's own drag-pan fights with our
      // rectangle-drawing on the same mousedown/mousemove.
      map.dragging.disable();

      drawing.current = true;

      setStart(e.latlng);
      setCurrent(e.latlng);
    },

    mousemove(e) {
      if (!drawing.current) {
        return;
      }

      setCurrent(e.latlng);
    },

    mouseup(e) {
      // Always re-enable dragging, even if this mouseup
      // doesn't correspond to an active drawing session.
      map.dragging.enable();

      if (!drawing.current || !start) {
        return;
      }

      drawing.current = false;

      const min_lat = Math.min(start.lat, e.latlng.lat);
      const max_lat = Math.max(start.lat, e.latlng.lat);
      const min_lon = Math.min(start.lng, e.latlng.lng);
      const max_lon = Math.max(start.lng, e.latlng.lng);

      if (
        Math.abs(max_lat - min_lat) < 0.001 ||
        Math.abs(max_lon - min_lon) < 0.001
      ) {
        setStart(null);
        setCurrent(null);
        return;
      }

      onSelect({ min_lon, min_lat, max_lon, max_lat });

      setStart(null);
      setCurrent(null);
    },
  });

  const bounds =
    start && current
      ? [
          [
            Math.min(start.lat, current.lat),
            Math.min(start.lng, current.lng),
          ],
          [
            Math.max(start.lat, current.lat),
            Math.max(start.lng, current.lng),
          ],
        ]
      : null;

  return bounds ? (
    <Rectangle
      bounds={bounds}
      pathOptions={{
        color: "#91d45d",
        weight: 3,
        fillOpacity: 0.2,
      }}
    />
  ) : null;
}

function App() {
  const [bbox, setBbox] = useState(null);
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [maxCloud, setMaxCloud] = useState(20);
  const [processing, setProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  // ==========================================================
  // MAP
  // ==========================================================

  const handleMapSelection = (newBbox) => {
    setBbox(newBbox);
    setResult(null);
    setError("");
  };

  // ==========================================================
  // SUPER RESOLUTION
  // ==========================================================

  const runSuperResolution = async () => {
    setError("");
    setResult(null);

    if (!bbox) {
      setError("Please click on the map to select an area.");
      return;
    }

    if (!startDate) {
      setError("Please select a start date.");
      return;
    }

    if (!endDate) {
      setError("Please select an end date.");
      return;
    }

    if (startDate > endDate) {
      setError("Start date cannot be after end date.");
      return;
    }

    const cloud = Number(maxCloud);

    if (Number.isNaN(cloud) || cloud < 0 || cloud > 100) {
      setError("Cloud cover must be between 0 and 100.");
      return;
    }

    setProcessing(true);

    try {
      const response = await fetch(
        `${API_URL}/api/super-resolve-area`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            min_lon: bbox.min_lon,
            min_lat: bbox.min_lat,
            max_lon: bbox.max_lon,
            max_lat: bbox.max_lat,
            start_date: startDate,
            end_date: endDate,
            max_cloud_coverage: cloud,
          }),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Super-resolution failed.");
      }

      setResult(data);
    } catch (err) {
      console.error(err);
      setError(err.message || "Super-resolution processing failed.");
    } finally {
      setProcessing(false);
    }
  };

  // ==========================================================
  // RESULT SCREEN
  // ==========================================================

  if (result) {
    const diagnostics = result.diagnostics;

    return (
      <div className="app">
        <section className="hero">
          <div className="hero-left">
            <div className="eyebrow">SENTINEL-2 · 10M → 2.5M</div>

            <h1>
              Sharpen satellite imagery
              <br />
              with AI.
            </h1>

            <p>
              Upload a compatible Sentinel-2 RGBN GeoTIFF and generate a
              4× super-resolved image using our fine-tuned neural
              network.
            </p>
          </div>

          <div className="hero-right"></div>
        </section>

        {/* =================================================
            RESULT HEADER
            ================================================= */}

        <section className="result-header">
          <div>
            <h2>Super Resolution Complete</h2>
            <p>
              Sentinel-2 L2A data processed for the selected Area of
              Interest.
            </p>
          </div>

          <button
            className="secondary-button"
            onClick={() => {
              setResult(null);
            }}
          >
            ← Process Another Area
          </button>
        </section>

        {/* =================================================
            IMAGE COMPARISON
            ================================================= */}

        <section className="comparison-grid">
          <div className="image-card">
            <h3>Original Sentinel-2</h3>
            <p className="image-meta">10m • B04 / B03 / B02</p>

            {result.original_preview && (
              <img src={result.original_preview} alt="Original Sentinel-2" />
            )}
          </div>

          <div className="image-card">
            <h3>AI Super-Resolved</h3>
            <p className="image-meta">2.5m • AI-derived output</p>

            {result.sr_preview && (
              <img
                src={result.sr_preview}
                alt="AI super-resolved Sentinel-2"
              />
            )}
          </div>
        </section>

        {/* =================================================
            RESULT INFO
            ================================================= */}

        <section className="result-info">
          <div className="info-card">
            <span>Input Resolution</span>
            <strong>{result.sentinel2?.resolution_m || 10}m</strong>
          </div>

          <div className="info-card">
            <span>Output Resolution</span>
            <strong>{result.output?.resolution_m || 2.5}m</strong>
          </div>

          <div className="info-card">
            <span>Model Device</span>
            <strong>{result.device || "Unknown"}</strong>
          </div>

          <div className="info-card">
            <span>Input Bands</span>
            <strong>B02 / B03 / B04 / B08</strong>
          </div>
        </section>

        {/* =================================================
            DIAGNOSTICS
            ================================================= */}

        {diagnostics && (
          <section className="diagnostic-section">
            <div className="diagnostic-title">DIAGNOSTIC INFORMATION</div>

            <div className="diagnostic-grid">
              <div className="diagnostic-card">
                <span>Input Min</span>
                <strong>
                  {diagnostics.input_global?.min?.toFixed(6)}
                </strong>
              </div>

              <div className="diagnostic-card">
                <span>Input Max</span>
                <strong>
                  {diagnostics.input_global?.max?.toFixed(6)}
                </strong>
              </div>

              <div className="diagnostic-card">
                <span>Input Mean</span>
                <strong>
                  {diagnostics.input_global?.mean?.toFixed(6)}
                </strong>
              </div>

              <div className="diagnostic-card">
                <span>Output Min</span>
                <strong>
                  {diagnostics.output_global?.min?.toFixed(6)}
                </strong>
              </div>

              <div className="diagnostic-card">
                <span>Output Max</span>
                <strong>
                  {diagnostics.output_global?.max?.toFixed(6)}
                </strong>
              </div>

              <div className="diagnostic-card">
                <span>Output Mean</span>
                <strong>
                  {diagnostics.output_global?.mean?.toFixed(6)}
                </strong>
              </div>

              <div className="diagnostic-card">
                <span>Normalization</span>
                <strong>
                  {diagnostics.normalization?.divided_by_10000
                    ? "DN → Reflectance"
                    : "Already Reflectance"}
                </strong>
              </div>

              <div className="diagnostic-card">
                <span>Output Clipped at 0</span>
                <strong>
                  {diagnostics.clipping?.clipped_low_percent?.toFixed(2)}%
                </strong>
              </div>
            </div>

            {/* =================================================
                BAND TABLE
                ================================================= */}

            <div className="diagnostic-table-wrapper">
              <h3>Normalized Model Input</h3>
              <table className="diagnostic-table">
                <thead>
                  <tr>
                    <th>Band</th>
                    <th>Min</th>
                    <th>Mean</th>
                    <th>P50</th>
                    <th>P99</th>
                    <th>Max</th>
                  </tr>
                </thead>

                <tbody>
                  {["B04", "B03", "B02", "B08"].map((band) => {
                    const stats = diagnostics.normalized_bands?.[band];

                    return (
                      <tr key={band}>
                        <td>{band}</td>
                        <td>{stats?.min?.toFixed(6)}</td>
                        <td>{stats?.mean?.toFixed(6)}</td>
                        <td>{stats?.p50?.toFixed(6)}</td>
                        <td>{stats?.p99?.toFixed(6)}</td>
                        <td>{stats?.max?.toFixed(6)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            <div className="diagnostic-table-wrapper">
              <h3>CNN Output</h3>
              <table className="diagnostic-table">
                <thead>
                  <tr>
                    <th>Band</th>
                    <th>Min</th>
                    <th>Mean</th>
                    <th>P50</th>
                    <th>P99</th>
                    <th>Max</th>
                  </tr>
                </thead>

                <tbody>
                  {["B04", "B03", "B02", "B08"].map((band) => {
                    const stats = diagnostics.output_bands?.[band];

                    return (
                      <tr key={band}>
                        <td>{band}</td>
                        <td>{stats?.min?.toFixed(6)}</td>
                        <td>{stats?.mean?.toFixed(6)}</td>
                        <td>{stats?.p50?.toFixed(6)}</td>
                        <td>{stats?.p99?.toFixed(6)}</td>
                        <td>{stats?.max?.toFixed(6)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </section>
        )}

        {/* =================================================
            DOWNLOAD
            ================================================= */}

        {result.output?.download_url && (
          <div className="download-section">
            <a
              className="download-button"
              href={`${API_URL}${result.output.download_url}`}
              target="_blank"
              rel="noreferrer"
            >
              Download 2.5m GeoTIFF
            </a>
          </div>
        )}
      </div>
    );
  }

  // ==========================================================
  // MAIN SCREEN
  // ==========================================================

  return (
    <div className="app">
      <HeroBackground />

      <header className="hero">
        <div className="hero-badge">SAT-SR • SIH 2026</div>

        <h1>
          Deep Learning Based
          <br />
          Super Resolution Mapping
        </h1>

        <p>
          Select an area on the map and generate AI-enhanced 2.5m
          Sentinel-2 imagery.
        </p>
      </header>

      <main className="container">
        {/* =================================================
            STEP 1
            ================================================= */}

        <section className="step-card">
          <div className="step-heading">
            <div className="step-number">1</div>

            <div>
              <h2>Select Area of Interest</h2>
              <p>Click on the map to select an area for processing.</p>
            </div>
          </div>

          <div className="map-wrapper">
            <MapContainer
              center={DEFAULT_CENTER}
              zoom={10}
              scrollWheelZoom={true}
              className="map"
            >
              <TileLayer
                attribution='&copy; OpenStreetMap contributors'
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              />

              <MapSelector onSelect={handleMapSelection} />

              {bbox && (
                <Rectangle
                  bounds={[
                    [bbox.min_lat, bbox.min_lon],
                    [bbox.max_lat, bbox.max_lon],
                  ]}
                />
              )}
            </MapContainer>
          </div>

          {bbox && (
            <div className="bbox-info">
              <span>Area Selected</span>

              <code>
                {bbox.min_lon.toFixed(5)}
                {" , "}
                {bbox.min_lat.toFixed(5)}
                {" → "}
                {bbox.max_lon.toFixed(5)}
                {" , "}
                {bbox.max_lat.toFixed(5)}
              </code>
            </div>
          )}
        </section>

        {/* =================================================
            STEP 2
            ================================================= */}

        <section className="step-card">
          <div className="step-heading">
            <div className="step-number">2</div>

            <div>
              <h2>Choose Image Conditions</h2>
              <p>
                Select the acquisition period and maximum cloud coverage.
              </p>
            </div>
          </div>

          <div className="controls-grid">
            <div className="control-group">
              <label>Start Date</label>

              <input
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
              />
            </div>

            <div className="control-group">
              <label>End Date</label>

              <input
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
              />
            </div>

            <div className="control-group">
              <label>Maximum Cloud Cover</label>

              <div className="cloud-input">
                <input
                  type="number"
                  min="0"
                  max="100"
                  value={maxCloud}
                  onChange={(e) => setMaxCloud(e.target.value)}
                />
                <span>%</span>
              </div>
            </div>
          </div>

          <button
            className="primary-button"
            onClick={runSuperResolution}
            disabled={processing}
          >
            {processing
              ? "Downloading + Processing..."
              : " Generate 2.5m Super-Resolved Image"}
          </button>
        </section>

        {/* =================================================
            ERROR
            ================================================= */}

        {error && <div className="error-box">{error}</div>}

        {/* =================================================
            FOOTER
            ================================================= */}

        <footer>
          <p>SAT-SR • Sentinel-2 Super Resolution</p>
          <p>10m input → 2.5m AI-derived output</p>
        </footer>
      </main>
    </div>
  );
}

export default App;