// client: docs update
// File: client/src/App.jsx

import { useState, useRef, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import { CIFAR10_LABELS } from "./lib/labels";
import "./index.css";

/** Root component: loads models, handles upload/webcam, runs human detection and CIFAR-10 classification. */
export default function App() {
  const [fileURL, setFileURL] = useState("");
  const [model, setModel] = useState(null);
  const [detModel, setDetModel] = useState(null);
  const [status, setStatus] = useState("Idle");
  const [isWarming, setIsWarming] = useState(false);
  const [preds, setPreds] = useState([]);
  const [human, setHuman] = useState({ present: false, score: 0 });
  const [error, setError] = useState("");

  const imgRef = useRef(null);
  const canvasRef = useRef(null);
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const rafIdRef = useRef(null);
  const lastTsRef = useRef(0);
  const humanRef = useRef({ present: false, score: 0 });
  useEffect(() => { humanRef.current = human; }, [human]);

  const [isLive, setIsLive] = useState(false);
  const detEveryN = useRef(6);
  const detTick = useRef(0);

  /** One-time setup: choose backend, load TF.js classifier and COCO-SSD detector, warm the classifier. */
  useEffect(() => {
    let isMounted = true;
    (async () => {
      try {
        setError("");
        const v = tf?.version?.tfjs ?? "unknown";
        setStatus(`Setting up backend (TFJS ${v})…`);
        try {
          await tf.setBackend("webgl");
          await tf.ready();
        } catch {
          setStatus(`WebGL failed; falling back to CPU (TFJS ${v})…`);
          await tf.setBackend("cpu");
          await tf.ready();
        }

        setStatus("Loading classifier…");
        const m = await tf.loadLayersModel(`/model/model.json?v=${Date.now()}`);
        setIsWarming(true);
        m.predict(tf.zeros([1, 32, 32, 3])).dispose();
        setIsWarming(false);
        if (!isMounted) return;
        setModel(m);
        setStatus("Classifier loaded ");

        setStatus("Loading person detector…");
        try {
          const dm = await cocoSsd.load({ base: "lite_mobilenet_v2" });
          if (isMounted) {
            setDetModel(dm);
            setStatus("Models loaded ");
          }
        } catch (e) {
          if (isMounted) setStatus("Classifier loaded (detector unavailable) ");
        }
      } catch (e) {
        if (isMounted) {
          setError(String(e?.message || e));
          setStatus("No model found yet (we'll add it later) ");
        }
      }
    })();
    return () => {
      isMounted = false;
      stopLiveInternal();
    };
  }, []);

  /** Draws the source into a 32×32 canvas using “contain” (letterbox). */
  function drawContain(ctx, srcEl, srcW, srcH) {
    const W = 32, H = 32;
    const scale = Math.min(W / srcW, H / srcH);
    const drawW = srcW * scale;
    const drawH = srcH * scale;
    const dx = (W - drawW) / 2;
    const dy = (H - drawH) / 2;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, W, H);
    ctx.drawImage(srcEl, dx, dy, drawW, drawH);
  }

  /** Draws the source into a 32×32 canvas using “cover” (center crop). */
  function drawCover(ctx, srcEl, srcW, srcH) {
    const W = 32, H = 32;
    const scale = Math.max(W / srcW, H / srcH);
    const drawW = srcW * scale;
    const drawH = srcH * scale;
    const dx = (W - drawW) / 2;
    const dy = (H - drawH) / 2;
    ctx.clearRect(0, 0, W, H);
    ctx.drawImage(srcEl, dx, dy, drawW, drawH);
  }

  /** Runs the classifier on the 32×32 canvas and returns top-2 predictions and max probability. */
  function classifyFromCanvas(canvas, model) {
    return tf.tidy(() => {
      const input = tf.browser.fromPixels(canvas).toFloat().expandDims(0);
      const out = model.predict(input);
      const probs = out.dataSync();
      const sorted = [...probs].map((p, i) => ({ i, p })).sort((a, b) => b.p - a.p);
      const top2 = sorted.slice(0, 2).map(({ i, p }) => ({
        label: CIFAR10_LABELS[i] ?? `class_${i}`,
        prob: p,
      }));
      const maxP = sorted[0]?.p ?? 0;
      return { top2, maxP };
    });
  }

  /** Handles file selection for still image prediction. */
  const handleFile = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setPreds([]);
    setHuman({ present: false, score: 0 });
    setError("");
    const url = URL.createObjectURL(file);
    setFileURL(url);
  };

  /** Detects human first; if none, classifies the still image using the better of contain/cover. */
  const predict = async () => {
    try {
      if (!model) { setStatus("Cannot predict: model not loaded"); return; }
      if (!imgRef.current || !canvasRef.current) { setStatus("Cannot predict: image/canvas missing"); return; }

      setStatus("Running inference…");
      setError("");

      if (detModel) {
        try {
          const dets = await detModel.detect(imgRef.current, 3);
          const person = dets.find((d) => d.class === "person");
          if (person) {
            setHuman({ present: true, score: person.score });
            setPreds([]);
            setStatus("Human detected ");
            return;
          } else {
            setHuman({ present: false, score: 0 });
          }
        } catch {}
      }

      const img = imgRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");
      canvas.width = 32; canvas.height = 32;

      drawContain(ctx, img, img.naturalWidth, img.naturalHeight);
      const resContain = classifyFromCanvas(canvas, model);

      drawCover(ctx, img, img.naturalWidth, img.naturalHeight);
      const resCover = classifyFromCanvas(canvas, model);

      const better = resContain.maxP >= resCover.maxP ? resContain : resCover;
      setPreds(better.top2);
      setStatus("Done ");
    } catch (e) {
      setError(String(e?.message || e));
      setStatus("Predict failed ");
    }
  };

  /** Starts webcam stream and enters a live loop with periodic human detection. */
  const startLive = async () => {
    try {
      if (!model) { setError("Load the model first."); return; }
      if (isLive) return;

      setError("");
      setPreds([]);
      setHuman({ present: false, score: 0 });
      setStatus("Requesting camera…");

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" }, width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      streamRef.current = stream;
      const video = videoRef.current;
      video.srcObject = stream;
      await video.play();

      setIsLive(true);
      setStatus("Live mode: running…");
      lastTsRef.current = 0;
      detTick.current = 0;
      rafIdRef.current = requestAnimationFrame(liveLoop);
    } catch (e) {
      setError(String(e?.message || e));
      setStatus("Live start failed ");
      stopLiveInternal();
    }
  };

  /** Public stop handler for live mode. */
  const stopLive = () => { stopLiveInternal(); setStatus("Live stopped ⏸"); };

  /** Tears down the webcam stream and animation loop, and clears results. */
  function stopLiveInternal() {
    if (rafIdRef.current) { cancelAnimationFrame(rafIdRef.current); rafIdRef.current = null; }
    if (streamRef.current) { streamRef.current.getTracks().forEach(t => t.stop()); streamRef.current = null; }
    if (isLive) setIsLive(false);
    setHuman({ present: false, score: 0 });
    setPreds([]);
  }

  /** Captures a single webcam frame and switches to still-image preview flow. */
  const captureToPreview = async () => {
    try {
      if (!isLive || !videoRef.current) { setError("Start Live first, then capture a frame."); return; }
      const v = videoRef.current;
      const w = v.videoWidth || 640, h = v.videoHeight || 480;

      const snap = document.createElement("canvas");
      snap.width = w; snap.height = h;
      const sctx = snap.getContext("2d");
      sctx.drawImage(v, 0, 0, w, h);

      await new Promise((resolve) => {
        snap.toBlob((blob) => {
          if (!blob) { setError("Could not capture a snapshot."); return resolve(); }
          const url = URL.createObjectURL(blob);
          stopLiveInternal();
          setFileURL(url);
          setPreds([]); setError("");
          setStatus("Captured frame → ready to Predict");
          resolve();
        }, "image/png", 0.92);
      });
    } catch (e) {
      setError(String(e?.message || e));
    }
  };

  /** Animation loop for live mode: throttles, hides predictions when a human is present, refreshes detection. */
  const liveLoop = (ts) => {
    if (!isLive || !videoRef.current || !canvasRef.current) return;

    const last = lastTsRef.current || 0;
    if (ts - last < 120) {
      rafIdRef.current = requestAnimationFrame(liveLoop);
      return;
    }
    lastTsRef.current = ts;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    canvas.width = 32; canvas.height = 32;

    drawContain(ctx, video, video.videoWidth, video.videoHeight);

    if (humanRef.current.present) {
      if (preds.length) setPreds([]);
    } else {
      try {
        const { top2 } = classifyFromCanvas(canvas, model);
        setPreds(top2);
      } catch (e) {
        setError(String(e?.message || e));
        stopLiveInternal();
        setStatus("Live failed ");
        return;
      }
    }

    detTick.current = (detTick.current + 1) % detEveryN.current;
    if (detTick.current === 0 && detModel) {
      detModel.detect(video, 3)
        .then(dets => {
          const person = dets.find(d => d.class === "person");
          setHuman({ present: !!person, score: person ? person.score : 0 });
        })
        .catch(() => {});
    }

    rafIdRef.current = requestAnimationFrame(liveLoop);
  };

  return (
    <>
      {error && (
        <div className="error"><strong>Error:</strong> {error}</div>
      )}

      <div className="header">
        <h1> Image Classifier </h1>
        <div className="badge">{isWarming ? "Warming…" : status}</div>
      </div>

      <div className="container">
        <div className="card">
          <h2>Choose input</h2>

          <div className="toolbar">
            <button className="btn" onClick={startLive} disabled={!model || isWarming || isLive}>
              {isLive ? "Live running…" : "Start Webcam"}
            </button>
            <button className="btn" onClick={captureToPreview} disabled={!isLive || isWarming}>
              Capture 
            </button>
            <button className="btn secondary" onClick={stopLive} disabled={!isLive}>
              Stop Live
            </button>
          </div>

          <div className="webcam" style={{ display: isLive ? "block" : "none" }}>
            <video ref={videoRef} autoPlay playsInline muted className="video" />
          </div>

          <div style={{ marginTop: 12 }}>
            <h3>Or upload a single image</h3>
            <input
              type="file"
              accept="image/*"
              onChange={handleFile}
              disabled={isLive || isWarming}
            />
          </div>

          {fileURL && !isLive && (
            <div className="preview">
              <img ref={imgRef} src={fileURL} alt="preview" />
              <div>
                <h3>Top-2 predictions</h3>
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

          <div style={{ marginTop: 12 }}>
            <div style={{ marginBottom: 8, opacity: 0.95 }}>
              <strong>Human:</strong>{" "}
              {human.present ? `Yes (${(human.score * 100).toFixed(1)}%)` : "No"}
            </div>

            {!human.present && preds.length ? (
              <ul style={{ listStyle: "none", padding: 0, margin: 0 }}>
                {preds.map((p, i) => (
                  <li key={i} className="prob">
                    {p.label} — {(p.prob * 100).toFixed(1)}%
                  </li>
                ))}
              </ul>
            ) : (!human.present ? <p>No predictions yet.</p> : null)}
          </div>

          <canvas ref={canvasRef} style={{ display: "none" }} />

          
        </div>
      </div>
    </>
  );
}