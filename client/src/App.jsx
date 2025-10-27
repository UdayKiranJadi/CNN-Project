import { useState } from "react";

export default function App() {
  const [fileURL, setFileURL] = useState("");

  const handleFile = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    // create a temporary URL so we can preview the image
    const url = URL.createObjectURL(file);
    setFileURL(url);
  };

  return (
    <div style={{ padding: 24, fontFamily: "system-ui, sans-serif" }}>
      <h1>Image Classifier (Step 2: Upload + Preview)</h1>

      <input type="file" accept="image/*" onChange={handleFile} />

      {fileURL && (
        <div style={{ marginTop: 16 }}>
          <p>Preview:</p>
          <img
            src={fileURL}
            alt="preview"
            style={{ width: 200, height: 200, objectFit: "contain", border: "1px solid #ccc", borderRadius: 8 }}
          />
        </div>
      )}
    </div>
  );
}
