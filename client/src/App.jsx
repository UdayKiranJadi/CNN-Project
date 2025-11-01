// File: client/src/App.jsx                      // main UI component
// Purpose: Upload image, load TF.js model, run prediction, show errors clearly

import { useState, useRef, useEffect } from "react";   // React hooks
import * as tf from "@tensorflow/tfjs";                // TensorFlow.js runtime
import { CIFAR10_LABELS } from "./lib/labels";         // CIFAR-10 class labels
import "./index.css";                                  // global styles

export default function App() {
  const [fileURL, setFileURL] = useState("");          // preview URL for uploaded image
  const [model, setModel] = useState(null);            // loaded TF.js model
  const [status, setStatus] = useState("Idle");        // human-friendly status text
  const [isWarming, setIsWarming] = useState(false);   // NEW: true while we run the warm-up pass
  const [preds, setPreds] = useState([]);              // top-5 predictions
  const [error, setError] = useState("");              // NEW: holds any error message to show in banner
  const canvasRef = useRef(null);                      // hidden canvas (32x32)
  const imgRef = useRef(null);                         // preview <img>

  useEffect(() => {
    let isMounted = true;                              // guard to avoid setState after unmount

    async function loadModel() {
      try {
        // Show TFJS version/backend right away (helps with debugging)
        const v = tf?.version_core || "unknown";
        setStatus(`Setting up backend (TFJS ${v})...`);
        setError("");                                  // CLEAR any previous error on retry

        // Try WebGL, fall back to CPU if needed
        try {
          await tf.setBackend("webgl");
          await tf.ready();
        } catch {
          setStatus(`WebGL failed; using CPU (TFJS ${v})...`);
          await tf.setBackend("cpu");
          await tf.ready();
        }

        // Candidate URLs for model.json (absolute + relative)
        const base = window.location.origin;
        const urls = ["/model/model.json", "model/model.json", `${base}/model/model.json`];

        // Try each until one works
        let lastErr = null;
        for (const url of urls) {
          try {
            setStatus(`Loading model from ${url} ...`);
            setError("");                              // clear banner before each attempt
            const m = await tf.loadLayersModel(url, {
              requestInit: { cache: "no-store", credentials: "omit", mode: "cors" }, // force no-cache
            });

            // Warm-up pass so first real inference is fast
            setIsWarming(true);                        // NEW: mark warming state
            setStatus("Warming up model...");
            m.predict(tf.zeros([1, 32, 32, 3])).dispose();
            setIsWarming(false);                       // NEW: warm-up finished

            if (isMounted) {
              setModel(m);                             // save model
              setStatus("Model loaded ✅");            // show success
            }
            return;                                    // done
          } catch (e) {
            lastErr = e;
            console.warn("Failed to load from", url, e);
          }
        }

        // If none worked, raise the final error
        throw lastErr || new Error("Model load failed for all candidates");
      } catch (err) {
        // Surface the exact reason both in console and in the banner
        console.error("Model load error:", err);
        if (isMounted) {
          setError(String(err?.message || err));       // NEW: show message in banner
          setStatus("No model found yet (we'll add it later) ⚠️");
        }
      }
    }

    loadModel();
    return () => { isMounted = false; };               // cleanup
  }, []);

  const handleFile = (e) => {
    const file = e.target.files?.[0];                  // first selected file
    if (!file) return;
    setPreds([]);                                      // clear previous predictions
    setError("");                                      // clear any previous error
    const url = URL.createObjectURL(file);             // correct API for preview URL
    setFileURL(url);                                   // show preview
  };

  const predict = () => {
    try {
      if (!model) {
        setStatus("Cannot predict: model not loaded");
        return;
      }
      if (!imgRef.current || !canvasRef.current) {
        setStatus("Cannot predict: image or canvas missing");
        return;
      }

      const img = imgRef.current;                      // <img> element
      const canvas = canvasRef.current;                // hidden <canvas>
      const ctx = canvas.getContext("2d");             // 2D context
      const W = 32, H = 32;                            // CIFAR-10 input size
      canvas.width = W;
      canvas.height = H;

      // Cover-fit the image into 32x32
      const scale = Math.max(W / img.naturalWidth, H / img.naturalHeight);
      const drawW = img.naturalWidth * scale;
      const drawH = img.naturalHeight * scale;
      const dx = (W - drawW) / 2;
      const dy = (H - drawH) / 2;

      ctx.clearRect(0, 0, W, H);
      ctx.drawImage(img, dx, dy, drawW, drawH);

      setStatus("Running inference...");
      setError("");                                    // clear any previous error

      const top5 = tf.tidy(() => {
        const input = tf.browser
          .fromPixels(canvas)                          // read pixels [32,32,3]
          .toFloat()                                   // u8 → f32
          .div(255)                                    // normalize to [0,1]
          .expandDims(0);                              // add batch dim: [1,32,32,3]

        const out = model.predict(input);              // forward pass → [1,10]
        const probs = out.dataSync();                  // read probabilities

        return [...probs]
          .map((p, i) => ({ i, p }))                   // pair prob with class
          .sort((a, b) => b.p - a.p)                   // sort high→low
          .slice(0, 5)                                 // top-5
          .map(({ i, p }) => ({                        // map to labels
            label: CIFAR10_LABELS[i] ?? `class_${i}`,
            prob: p
          }));
      });

      setPreds(top5);                                  // update predictions
      setStatus("Done ✅");                            // show done
    } catch (err) {
      console.error("Predict error:", err);            // print for debugging
      setError(String(err?.message || err));           // show in banner
      setStatus("Predict failed ❌");                  // update status
    }
  };

  return (
    <>
      {/* ERROR BANNER (only shows when error text is non-empty) */}
      {error && (
        <div className="error">                        {/* simple red banner */}
          <strong>Error:</strong> {error}
        </div>
      )}

      <div className="header">
        <h1>Image Classifier (React + TF.js)</h1>
        {/* Badge shows live status, including warm-up */}
        <div className="badge">
          {isWarming ? "Warming…" : status}
        </div>
      </div>

      <div className="container">
        <div className="card">
          <h2>Upload image</h2>
          <input
            type="file"
            accept="image/*"
            onChange={handleFile}                      // file input handler
            disabled={isWarming}                       // block upload during warm-up
          />

          {fileURL && (
            <div className="preview">
              <img ref={imgRef} src={fileURL} alt="preview" />
              <div>
                <h3>Predictions</h3>
                <button
                  onClick={predict}
                  disabled={!model || isWarming}       // disable while warming or no model
                  className="btn"
                  style={{ padding: "8px 12px", borderRadius: 8, marginBottom: 12 }}
                >
                  {model ? (isWarming ? "Warming…" : "Predict") : "Model not loaded"}
                </button>

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
            </div>
          )}

          {/* Hidden canvas used for 32x32 resizing before prediction */}
          <canvas ref={canvasRef} style={{ display: "none" }} />
          <p style={{ opacity: 0.7, marginTop: 12 }}>
            CIFAR-10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
          </p>
        </div>
      </div>
    </>
  );
}
