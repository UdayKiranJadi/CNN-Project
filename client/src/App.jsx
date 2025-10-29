import { useState, useRef } from "react";
import "./index.css";

export default function App() {
  const [fileURL, setFileURL] = useState("");
  const imgRef = useRef(null);

  const handleFile = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setFileURL(URL.createObjectURL(file));
  };

  return (
    <>
    <div className = "header">
      <h1>Image Classifier (react) </h1>
      <div classname="badge">step 3: ui changes</div>

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
              <p>comming next : tesnorflow.js inference</p>
              </div>
              </div>
        )}
      </div>
    </div>
    </>

    
    
  );
}
