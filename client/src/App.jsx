import { useState, useRef, useEffect} from "react";
import * as tf from "@tensorflow/tfjs";
import {CIFAR10_LABELS} from "./lib/labels";
import "./index.css";

export default function App() {
  const [fileURL, setFileURL] = useState("");
  const [model, setModel] = useState(null);
  const [status, setStatus] = useState("Idle");
  const [preds, setPreds] = useState([]);
  const canvasRef = useRef(null);
  const imgRef = useRef(null);


  useEffect(() => {
    let isMounted = true;
    (async () => {
      try {
        setStatus("Setting up backend.....");
        await tf.setBackend("webgl");
        await tf.ready();

        setStatus("Loading model (if present)...");
        const m = await tf.loadLayerModel("/model/model.json");
        setStatus("Warming model....");
        m.predict(tf.zeros([1,32,32, 3])).dispose();
        if (isMounted) {
          setModel(m);
          setStatus("Model  Loaded");
        }

      } catch (err){
        console.warn("Model not loaded yet: ", err?.message);
        if (isMounted) setStatus("no model found we will add it later");

      }
    }) ();
    return () => { isMounted = false; };

  },[]);

  const handleFile = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setPreds([]);
    const url = URL.createObjectsURL(file);
    setFileURL(url);
  };

  const predict = () => {
    if (!model) {
      setStatus("Cannot predict : model not Loaded");
      return;
    }
    if (!imgRef.current || !canvasRef.current) {
      setStatus("cannot predidct : img or canvas missing");
      return;
    }

    const img = imgRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const W = 32, H = 32;
    canvas.width = W;
    canvas.height = H;

    const scale = Math.max(W / img.naturalWidth, H / img.naturalHeight);
    const drawW = img.naturalWidth * scale;
    const drawH = img.naturalHeight * scale;
    const dx = (W - drawW) / 2;
    const dy = (H - drawH) / 2;

    ctx.clearRect(0, 0, W, H);
    ctx.drawImage(img, dx, dy, drawW, drawH);

    setStatus("running inference....");
    const top5 = tf.tidy(() => {
      const input = tf.browser
      .fromPixels(canvas)
      .toFloat()
      .div(255)
      .expandDims(0);

      const out = model.predict(input);
      const probs = out.datasync();

      return [...probs]
      .map((p, i) => ({i, p}))
      .sort((a, b) => b.p -a.p)
      .slice(0, 5)
      .map(({i,p}) => ({
        label: CIFAR10_LABELS[i] ?? `class_${i}`,
        prob: p
      }));
    });

    setPreds(top5);
    setStatus("done");
  };

  return (
    <>
    <div className = "header">
      <h1>Image Classifier (React + TF.js) </h1> {/* app title  */}
      <div classname="badge">{status}</div> {/* show current status to tje user */}

    </div>
    <div className = "container">
      <div className="card">
        <h2>upload image</h2>
        <input type="file" accept="image/*" onchange={handleFile}/>
        {fileURL && (
          <div className="preview">
            <img ref={imgRef} src={fileURL} alt = "preview"/>
            <div>
              <h3>Predictions</h3>
              <button 
              onClick = {predict}
              disabled= {!model}
              className="btn"
              style={{ padding : "8px 12px ", borderRadius: 8, marginBotton : 12}}
              >
                {model ? "Predict" : "Model not loded"}
              </button>

              {preds.length ? (
                <ul style = {{ listStyle: "none", padding: 0, marigin: 0}}>
                {preds.map((p, i) => (
                  <li key = {i} className="prob">
                    {p.label} - {p.prob.toFixed(4)}
                  </li>
                ))}
                
                
                </ul>
              ) : (
              
              <p> no Predictions yet </p>
              )}
              </div>
              </div>
        )}
        <canvas ref={canvasRef} style={{ display: "none" }} />
          <p style={{ opacity: 0.7, marginTop: 12 }}>
            CIFAR-10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
          </p>
      </div>
    </div>
    </>

    
    
  );
}
