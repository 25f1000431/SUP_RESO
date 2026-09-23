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
  const [showSegmentation, setShowSegmentation] = useState(false);

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

       <HeroBackground />
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
            <p className="image-meta">10m • B04 / B03 / B02 / B08</p>

            {result.original_preview && (
              <img src={result.original_preview} alt="Original Sentinel-2" />
            )}
          </div>

          <div className="image-card">
            <h3>AI Super-Resolved</h3>
            <p className="image-meta">2.5m GSD • B02 / B03 / B04 / B08 Super Resolved output </p>

            {result.sr_preview && (
              <img
                src={result.sr_preview}
                alt="AI super-resolved Sentinel-2"
              />
            )}
          </div>
        </section>

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




        {/* =================================================
            SEGMENTATION ANALYSIS
            ================================================= */}

        {result.segmentation?.status === "completed" && (
          <>
            {/* =================================================
                VIEW / HIDE SEGMENTATION BUTTON
                ================================================= */}

            <div className="segmentation-toggle-wrapper">
              <button
                type="button"
                className="segmentation-toggle"
                onClick={() => setShowSegmentation((prev) => !prev)}
              >
                {showSegmentation
                  ? "Hide Segmentation Analysis"
                  : "View Segmentation Analysis"}
              </button>
            </div>


            {/* =================================================
                SEGMENTATION ANALYSIS CONTENT
                ================================================= */}

            {showSegmentation && (
              <section className="segmentation-section">

                <div className="segmentation-header">

                  <div>
                    <div className="segmentation-eyebrow">
                      DEEPLABV3+ · RESNET34
                    </div>

                    <h2>Segmentation Analysis</h2>

                    <p>
                      Model-derived land-cover and vegetation analysis
                      from the AI super-resolved imagery.
                    </p>
                  </div>

                  <div className="segmentation-status">
                    ● ANALYSIS COMPLETE
                  </div>

                </div>


                {/* =================================================
                    SEGMENTATION SUMMARY CARDS
                    ================================================= */}

                <div className="segmentation-cards">

                  <div className="segmentation-card">
                    <span>CROPLAND</span>

                    <strong>
                      {result.segmentation.classes?.cropland?.percentage?.toFixed(2)}%
                    </strong>

                    <small>
                      {result.segmentation.classes?.cropland?.area_hectares?.toFixed(2)} ha
                    </small>
                  </div>


                  <div className="segmentation-card">
                    <span>LANDSLIDE</span>

                    <strong>
                      {result.segmentation.classes?.landslide?.percentage?.toFixed(2)}%
                    </strong>

                    <small>
                      {result.segmentation.classes?.landslide?.area_hectares?.toFixed(2)} ha
                    </small>
                  </div>


                  <div className="segmentation-card">
                    <span>MEAN CROP NDVI</span>

                    <strong>
                      {result.segmentation.crop_health?.mean_ndvi?.toFixed(4)}
                    </strong>

                    <small>
                      Median {result.segmentation.crop_health?.median_ndvi?.toFixed(4)}
                    </small>
                  </div>


                  <div className="segmentation-card">
                    <span>CROP HEALTH INDEX</span>

                    <strong>
                      {result.segmentation.crop_health?.health_score?.toFixed(0)}
                    </strong>

                    <small>
                      {result.segmentation.crop_health?.health_label}
                    </small>
                  </div>

                </div>


                {/* =================================================
                    VISUAL ANALYSIS MAPS
                    ================================================= */}

                <div className="analysis-image-grid">

                  {/* Segmentation Map */}

                  <div className="analysis-image-card">

                    <div className="analysis-card-header">

                      <div>
                        <h3>Cropland Segmentation</h3>
                        <span>
                          DeepLabV3+ · ResNet34
                        </span>
                      </div>

                    </div>

                    {result.segmentation?.preview_url && (
                      <img
                        src={`${API_URL}${result.segmentation.preview_url}`}
                        alt="Cropland segmentation map"
                        className="analysis-map-image"
                      />
                    )}

                  </div>


                  {/* NDVI Map */}

                  <div className="analysis-image-card">

                    <div className="analysis-card-header">

                      <div>
                        <h3>Crop NDVI Map</h3>
                        <span>
                          Vegetation index
                        </span>
                      </div>

                    </div>

                    {result.segmentation?.ndvi_preview_url && (
                      <img
                        src={`${API_URL}${result.segmentation.ndvi_preview_url}`}
                        alt="Crop NDVI map"
                        className="analysis-map-image"
                      />
                    )}

                  </div>


                  {/* Crop Health Map */}

                  <div className="analysis-image-card">

                    <div className="analysis-card-header">

                      <div>
                        <h3>Crop Health Index</h3>

                        <span>
                          {result.segmentation?.crop_health?.health_score != null
                            ? `${result.segmentation.crop_health.health_score.toFixed(1)}/100`
                            : "Health unavailable"}

                          {result.segmentation?.crop_health?.health_label
                            ? ` · ${result.segmentation.crop_health.health_label}`
                            : ""}
                        </span>

                      </div>

                    </div>

                    {result.segmentation?.health_preview_url && (
                      <img
                        src={`${API_URL}${result.segmentation.health_preview_url}`}
                        alt="Crop health index map"
                        className="analysis-map-image"
                      />
                    )}

                  </div>

                </div>


                {/* =================================================
                    CROP HEALTH + VEGETATION VIGOR
                    ================================================= */}

                <div className="analysis-grid">

                  {/* Crop Health */}

                  <div className="analysis-card">

                    <div className="analysis-card-header">
                      <h3>Crop Health</h3>
                      <span>NDVI-based</span>
                    </div>


                    <div className="health-score">

                      <strong>
                        {result.segmentation.crop_health?.health_score?.toFixed(0)}
                      </strong>

                      <span>/ 100</span>

                    </div>


                    <div className="health-label">
                      {result.segmentation.crop_health?.health_label}
                    </div>


                    <div className="health-details">

                      <div>
                        <span>Mean NDVI</span>

                        <strong>
                          {result.segmentation.crop_health?.mean_ndvi?.toFixed(4)}
                        </strong>
                      </div>


                      <div>
                        <span>Min NDVI</span>

                        <strong>
                          {result.segmentation.crop_health?.min_ndvi?.toFixed(4)}
                        </strong>
                      </div>


                      <div>
                        <span>Max NDVI</span>

                        <strong>
                          {result.segmentation.crop_health?.max_ndvi?.toFixed(4)}
                        </strong>
                      </div>


                      <div>
                        <span>NDVI Std.</span>

                        <strong>
                          {result.segmentation.crop_health?.std_ndvi?.toFixed(4)}
                        </strong>
                      </div>

                    </div>

                  </div>


                  {/* Vegetation Vigor Distribution */}

                  <div className="analysis-card">

                    <div className="analysis-card-header">
                      <h3>Vegetation Vigor Distribution</h3>
                      <span>Crop pixels</span>
                    </div>


                    <div className="vigor-list">

                      {/* Very Low */}

                      <div className="vigor-row">

                        <span>Very Low</span>

                        <div className="vigor-bar">
                          <div
                            className="vigor-fill"
                            style={{
                              width: `${result.segmentation.vigor_distribution?.very_low || 0}%`,
                            }}
                          />
                        </div>

                        <strong>
                          {result.segmentation.vigor_distribution?.very_low?.toFixed(2)}%
                        </strong>

                      </div>


                      {/* Low */}

                      <div className="vigor-row">

                        <span>Low</span>

                        <div className="vigor-bar">
                          <div
                            className="vigor-fill"
                            style={{
                              width: `${result.segmentation.vigor_distribution?.low || 0}%`,
                            }}
                          />
                        </div>

                        <strong>
                          {result.segmentation.vigor_distribution?.low?.toFixed(2)}%
                        </strong>

                      </div>


                      {/* Moderate */}

                      <div className="vigor-row">

                        <span>Moderate</span>

                        <div className="vigor-bar">
                          <div
                            className="vigor-fill"
                            style={{
                              width: `${result.segmentation.vigor_distribution?.moderate || 0}%`,
                            }}
                          />
                        </div>

                        <strong>
                          {result.segmentation.vigor_distribution?.moderate?.toFixed(2)}%
                        </strong>

                      </div>


                      {/* High */}

                      <div className="vigor-row">

                        <span>High</span>

                        <div className="vigor-bar">
                          <div
                            className="vigor-fill"
                            style={{
                              width: `${result.segmentation.vigor_distribution?.high || 0}%`,
                            }}
                          />
                        </div>

                        <strong>
                          {result.segmentation.vigor_distribution?.high?.toFixed(2)}%
                        </strong>

                      </div>


                      {/* Very High */}

                      <div className="vigor-row">

                        <span>Very High</span>

                        <div className="vigor-bar">
                          <div
                            className="vigor-fill"
                            style={{
                              width: `${result.segmentation.vigor_distribution?.very_high || 0}%`,
                            }}
                          />
                        </div>

                        <strong>
                          {result.segmentation.vigor_distribution?.very_high?.toFixed(2)}%
                        </strong>

                      </div>

                    </div>

                  </div>

                </div>


                {/* =================================================
                    SEGMENTATION DOWNLOAD
                    ================================================= */}

                <div className="segmentation-downloads">

                  <a
                    className="download-button"
                    href={`${API_URL}${result.segmentation.download_url}`}
                    target="_blank"
                    rel="noreferrer"
                  >
                    Download Segmentation GeoTIFF
                  </a>

                  {result.segmentation.ndvi_path && (
                    <span className="analysis-note">
                      NDVI analysis generated from the 4-band SR output.
                    </span>
                  )}

                </div>

              </section>
            )}

          </>
        )}
        

        
        {/* =================================================
            DOWNLOAD
            ================================================= */}


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