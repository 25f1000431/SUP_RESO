import { useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

function App() {
  const [file, setFile] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const handleFile = (selectedFile) => {
    setError("");
    setResult(null);

    if (!selectedFile) return;

    const valid =
      selectedFile.name.toLowerCase().endsWith(".tif") ||
      selectedFile.name.toLowerCase().endsWith(".tiff");

    if (!valid) {
      setError(
        "Please upload a Sentinel-2 GeoTIFF (.tif or .tiff)."
      );
      return;
    }

    setFile(selectedFile);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setDragging(false);

    const droppedFile =
      event.dataTransfer.files[0];

    handleFile(droppedFile);
  };

  const runSuperResolution = async () => {
    if (!file) return;

    setLoading(true);
    setError("");
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(
        `${API_URL}/api/super-resolve`,
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Super-resolution failed."
        );
      }

      setResult(data);

    } catch (err) {
      setError(
        err.message ||
        "Could not connect to the backend."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">

      <header className="header">

        <div>
          <div className="brand">
            SAT<span>-</span>SR
          </div>

          <div className="subtitle">
            AI Satellite Super-Resolution
          </div>
        </div>

        <div className="status">
          <span className="status-dot"></span>
          LOCAL AI ENGINE
        </div>

      </header>


      <main className="main">

        <section className="hero">

          <div className="eyebrow">
            SENTINEL-2 · 10M → 2.5M
          </div>

          <h1>
            Sharpen satellite imagery
            <br />
            with AI.
          </h1>

          <p>
            Upload a compatible Sentinel-2 RGBN
            GeoTIFF and generate a 4× super-resolved
            image using our fine-tuned neural network.
          </p>

        </section>


        <section className="upload-card">

          <div
            className={`drop-zone ${
              dragging ? "dragging" : ""
            }`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
          >

            <input
              id="fileInput"
              type="file"
              accept=".tif,.tiff"
              hidden
              onChange={(e) =>
                handleFile(
                  e.target.files[0]
                )
              }
            />

            <div className="upload-icon">
              ↑
            </div>

            <h2>
              {file
                ? file.name
                : "Upload Sentinel-2 imagery"}
            </h2>

            <p>
              Drag & drop a GeoTIFF here
              <br />
              or
            </p>

            <label
              htmlFor="fileInput"
              className="browse-button"
            >
              Browse files
            </label>

            <div className="requirements">
              B02 · B03 · B04 · B08
              <span>•</span>
              10m
              <span>•</span>
              GeoTIFF
            </div>

          </div>


          {file && (

            <button
              className="run-button"
              onClick={runSuperResolution}
              disabled={loading}
            >
              {loading
                ? "Running AI Super-Resolution..."
                : "Generate 2.5m Image →"}
            </button>

          )}

        </section>


        {error && (

          <div className="error-box">
            <strong>Upload rejected</strong>
            <div>{error}</div>
          </div>

        )}


        {result && (

          <section className="results">

            <div className="results-header">

              <div>
                <div className="eyebrow">
                  PROCESSING COMPLETE
                </div>

                <h2>
                  10m → 2.5m
                </h2>
              </div>

              <div className="device">
                Running on{" "}
                <strong>
                  {result.device}
                </strong>
              </div>

            </div>


            <div className="comparison">

              <div className="image-card">

                <div className="image-label">
                  ORIGINAL
                </div>

                <img
                  src={
                    result.original_preview
                  }
                  alt="Original Sentinel-2"
                />

                <div className="image-info">
                  {result.input.width} ×{" "}
                  {result.input.height}
                  {" · "}
                  10m
                </div>

              </div>


              <div className="image-card featured">

                <div className="image-label">
                  AI SUPER-RESOLVED
                </div>

                <img
                  src={
                    result.sr_preview
                  }
                  alt="AI super-resolved"
                />

                <div className="image-info">
                  {result.output.width} ×{" "}
                  {result.output.height}
                  {" · "}
                  2.5m
                </div>

              </div>

            </div>


            <a
              className="download-button"
              href={`${API_URL}${result.output.download_url}`}
              download
            >
              Download 2.5m GeoTIFF ↓
            </a>

          </section>

        )}

      </main>


      <footer>
        SAT-SR · Deep Learning Based
        Satellite Image Super-Resolution
      </footer>

    </div>
  );
}

export default App;