// File: client/src/App.jsx                       // Main UI component
// Purpose: Upload image OR use webcam "Live mode", load TF.js model, predict continuously.

import { useState, useRef, useEffect } from "react";  // React hooks for state/refs/effects
import * as tf from "@tensorflow/tfjs";               // TensorFlow.js runtime in the browser
import { CIFAR10_LABELS } from "./lib/labels";        // CIFAR-10 label names
import "./index.css";                                 // Global styles (header, banner, etc.)

export default function App() {
  // ---------- App state ----------
  const [fileURL, setFileURL] = useState("");         // Blob URL for still-image preview
  const [model, setModel] = useState(null);           // Loaded TF.js model
  const [status, setStatus] = useState("Idle");       // Human-readable status badge text
  const [isWarming, setIsWarming] = useState(false);  // True while running warm-up pass
  const [preds, setPreds] = useState([]);             // Top-5 predictions [{label, prob}]
  const [error, setError] = useState("");             // Error banner text (empty = hidden)

  // ---------- DOM refs ----------
  const canvasRef = useRef(null);                     // Hidden <canvas> used to resize to 32x32
  const imgRef = useRef(null);                        // <img> preview for still uploads
  const videoRef = useRef(null);                      // <video> for live webcam preview

  // ---------- Live mode control ----------
  const [isLive, setIsLive] = useState(false);        // True when webcam live mode is active
  const streamRef = useRef(null);                     // Holds MediaStream so we can stop tracks
  const rafIdRef = useRef(null);                      // requestAnimationFrame id for the loop
  const lastTsRef = useRef(0);                        // Timestamp to throttle FPS

  // ---------- Model loader (with fallbacks & error banner) ----------
  useEffect(() => {
    let isMounted = true;                             // Guard to avoid setState after unmount

    async function loadModel() {
      try {
        const v = tf?.version_core || "unknown";      // Log TFJS version for debugging
        setStatus(`Setting up backend (TFJS ${v})...`);
        setError("");                                 // Clear any old error

        // Prefer WebGL; fall back to CPU if it fails (browser/driver issues)
        try {
          await tf.setBackend("webgl");
          await tf.ready();
        } catch {
          setStatus(`WebGL failed; using CPU (TFJS ${v})...`);
          await tf.setBackend("cpu");
          await tf.ready();
        }

        // Try a few URL candidates in case path/base changes
        const base = window.location.origin;
        const urls = ["/model/model.json", "model/model.json", `${base}/model/model.json`];

        let lastErr = null;
        for (const url of urls) {
          try {
            setStatus(`Loading model from ${url} ...`);
            setError("");                             // Clear banner before each attempt
            const m = await tf.loadLayersModel(url, {
              requestInit: { cache: "no-store", credentials: "omit", mode: "cors" }, // no-cache
            });

            // Warm-up pass so first real inference is fast
            setIsWarming(true);
            setStatus("Warming up model...");
            m.predict(tf.zeros([1, 32, 32, 3])).dispose(); // 1 dummy forward
            setIsWarming(false);

            if (isMounted) {
              setModel(m);
              setStatus("Model loaded ✅");
            }
            return;                                   // Success → stop trying others
          } catch (e) {
            lastErr = e;
            console.warn("Failed to load from", url, e);
          }
        }

        // Nothing worked → throw to banner
        throw lastErr || new Error("Model load failed for all candidates");
      } catch (err) {
        console.error("Model load error:", err);
        if (isMounted) {
          setError(String(err?.message || err));      // Show exact error in banner
          setStatus("No model found yet (we'll add it later) ⚠️");
        }
      }
    }

    loadModel();
    return () => {                                    // Cleanup on unmount
      isMounted = false;
      stopLiveInternal();                              // Ensure live mode is fully stopped
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);                                             // Run once on mount

  // ---------- File input handler ----------
  const handleFile = (e) => {
    const file = e.target.files?.[0];                 // First selected file
    if (!file) return;
    setPreds([]);                                     // Clear old predictions
    setError("");                                     // Clear error banner
    const url = URL.createObjectURL(file);            // Correct API to create preview URL
    setFileURL(url);                                  // Trigger <img> to render
  };

  // ---------- One-shot predict for still image ----------
  const predict = () => {
    try {
      if (!model) { setStatus("Cannot predict: model not loaded"); return; }
      if (!imgRef.current || !canvasRef.current) { setStatus("Cannot predict: image or canvas missing"); return; }

      const img = imgRef.current;                     // Source is the <img> element
      const canvas = canvasRef.current;               // Draw into hidden canvas at 32x32
      const ctx = canvas.getContext("2d");
      const W = 32, H = 32;
      canvas.width = W; canvas.height = H;

      // Cover-fit (fill 32x32 without distortion)
      const scale = Math.max(W / img.naturalWidth, H / img.naturalHeight);
      const drawW = img.naturalWidth * scale;
      const drawH = img.naturalHeight * scale;
      const dx = (W - drawW) / 2;
      const dy = (H - drawH) / 2;

      ctx.clearRect(0, 0, W, H);
      ctx.drawImage(img, dx, dy, drawW, drawH);

      setStatus("Running inference...");
      setError("");

      const top5 = tf.tidy(() => {                    // Auto-clean intermediate tensors
        const input = tf.browser
          .fromPixels(canvas)                         // [32,32,3]
          .toFloat()                                  // uint8 → float32
          .div(255)                                   // Normalize [0,1]
          .expandDims(0);                             // [1,32,32,3]

        const out = model.predict(input);             // Forward pass
        const probs = out.dataSync();                 // Read probabilities

        return [...probs]
          .map((p, i) => ({ i, p }))
          .sort((a, b) => b.p - a.p)
          .slice(0, 5)
          .map(({ i, p }) => ({ label: CIFAR10_LABELS[i] ?? `class_${i}`, prob: p }));
      });

      setPreds(top5);
      setStatus("Done ✅");
    } catch (err) {
      console.error("Predict error:", err);
      setError(String(err?.message || err));
      setStatus("Predict failed ❌");
    }
  };

  // ---------- Live mode: start webcam + loop ----------
  const startLive = async () => {
    try {
      if (!model) { setError("Load the model first."); return; }
      if (isLive) return;                              // If already live, ignore

      setError("");                                    // Clear errors
      setPreds([]);                                    // Reset predictions
      setStatus("Requesting camera…");

      // Ask for webcam; prefer back camera if available
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" }, width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });

      streamRef.current = stream;                      // Save stream so we can stop tracks
      const video = videoRef.current;                  // Our <video> element
      video.srcObject = stream;                        // Attach stream to video
      await video.play();                              // Start playback
      setIsLive(true);                                 // Flip live flag
      setStatus("Live mode: running…");

      // Kick off the rAF loop
      lastTsRef.current = 0;                           // Reset throttle timer
      rafIdRef.current = requestAnimationFrame(liveLoop);
    } catch (err) {
      console.error("Live start error:", err);
      setError(String(err?.message || err));
      setStatus("Live start failed ❌");
      stopLiveInternal();                              // Ensure cleanup if partially started
    }
  };

  // ---------- Live mode: stop webcam + loop ----------
  const stopLive = () => {
    stopLiveInternal();                                // Centralized cleanup
    setStatus("Live stopped ⏸");
  };

  // Internal: shared cleanup (used by stopLive and unmount)
  const stopLiveInternal = () => {
    // Cancel animation frame loop
    if (rafIdRef.current) {
      cancelAnimationFrame(rafIdRef.current);
      rafIdRef.current = null;
    }
    // Stop camera tracks
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    // Reset live flag
    if (isLive) setIsLive(false);
  };

  // ---------- The live prediction loop (throttled) ----------
  const liveLoop = (ts) => {
    // If not live or no DOM ready, bail
    if (!isLive || !model || !videoRef.current || !canvasRef.current) return;

    // Throttle to ~8–12 FPS (here: run if ≥120ms elapsed)
    const last = lastTsRef.current || 0;
    if (ts - last < 120) {
      rafIdRef.current = requestAnimationFrame(liveLoop);
      return;
    }
    lastTsRef.current = ts;

    const video = videoRef.current;                    // Source: <video>
    const canvas = canvasRef.current;                  // Resize target: hidden <canvas>
    const ctx = canvas.getContext("2d");
    const W = 32, H = 32;
    canvas.width = W; canvas.height = H;

    // Cover-fit video frame into 32x32 (use videoWidth/Height)
    const scale = Math.max(W / video.videoWidth, H / video.videoHeight);
    const drawW = video.videoWidth * scale;
    const drawH = video.videoHeight * scale;
    const dx = (W - drawW) / 2;
    const dy = (H - drawH) / 2;

    // Draw one frame onto the canvas
    ctx.clearRect(0, 0, W, H);
    ctx.drawImage(video, dx, dy, drawW, drawH);

    // Predict (keep it small to avoid GC pressure)
    try {
      const top5 = tf.tidy(() => {
        const input = tf.browser.fromPixels(canvas).toFloat().div(255).expandDims(0);
        const out = model.predict(input);
        const probs = out.dataSync();
        return [...probs]
          .map((p, i) => ({ i, p }))
          .sort((a, b) => b.p - a.p)
          .slice(0, 5)
          .map(({ i, p }) => ({ label: CIFAR10_LABELS[i] ?? `class_${i}`, prob: p }));
      });
      setPreds(top5);                                  // Update UI with latest top-5
    } catch (err) {
      console.error("Live predict error:", err);
      setError(String(err?.message || err));
      stopLiveInternal();                              // Stop live to prevent runaway errors
      setStatus("Live failed ❌");
      return;
    }

    // Schedule next frame
    rafIdRef.current = requestAnimationFrame(liveLoop);
  };

  // ---------- UI ----------
  return (
    <>
      {/* ERROR BANNER */}
      {error && (
        <div className="error">
          <strong>Error:</strong> {error}
        </div>
      )}

      <div className="header">
        <h1>Image Classifier (React + TF.js)</h1>
        <div className="badge">{isWarming ? "Warming…" : status}</div>
      </div>

      <div className="container">
        <div className="card">
          <h2>Choose input</h2>

          {/* Toolbar: still-image vs webcam */}
          <div className="toolbar">
            <button
              className="btn"
              onClick={startLive}                       // Start webcam live mode
              disabled={!model || isWarming || isLive} // Need model ready, not warming, not already live
            >
              {isLive ? "Live running…" : "Start Live (webcam)"}
            </button>

            <button
              className="btn secondary"
              onClick={stopLive}                        // Stop webcam live
              disabled={!isLive}
            >
              Stop Live
            </button>
          </div>

          {/* Webcam preview (hidden until live) */}
          <div className="webcam" style={{ display: isLive ? "block" : "none" }}>
            <video
              ref={videoRef}                            // Bind the <video> element
              autoPlay
              playsInline                               // Avoid fullscreen on iOS
              muted                                     // No audio needed
              className="video"                         // Simple styling box
            />
          </div>

          {/* Still image upload (disabled while live) */}
          <div style={{ marginTop: 12 }}>
            <h3>Or upload a single image</h3>
            <input
              type="file"
              accept="image/*"
              onChange={handleFile}
              disabled={isLive || isWarming}            // Disable while live/warming
            />
          </div>

          {/* Preview + Predict for still image */}
          {fileURL && !isLive && (
            <div className="preview">
              <img ref={imgRef} src={fileURL} alt="preview" />
              <div>
                <h3>Predictions</h3>
                <button
                  onClick={predict}
                  disabled={!model || isWarming}
                  className="btn"
                  style={{ padding: "8px 12px", borderRadius: 8, marginBottom: 12 }}
                >
                  {model ? (isWarming ? "Warming…" : "Predict") : "Model not loaded"}
                </button>
              </div>
            </div>
          )}

          {/* Shared predictions list (works for both live & still) */}
          <div style={{ marginTop: 12 }}>
            {preds.length ? (
              <ul style={{ listStyle: "none", padding: 0, margin: 0 }}>
                {preds.map((p, i) => (
                  <li key={i} className="prob">
                    {p.label} — {p.prob.toFixed(4)}
                  </li>
                ))}
              </ul>
            ) : (
              <p>No predictions yet.</p>
            )}
          </div>

          {/* Hidden canvas used to resize to 32x32 for BOTH still + live */}
          <canvas ref={canvasRef} style={{ display: "none" }} />

          <p style={{ opacity: 0.7, marginTop: 12 }}>
            CIFAR-10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
          </p>
        </div>
      </div>
    </>
  );
}
