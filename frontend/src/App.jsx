import { useState } from "react";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];

    setFile(selected);

    if (selected) {
      setPreview(URL.createObjectURL(selected));
    }
  };

  const uploadImage = async () => {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch("http://127.0.0.1:8000/upload", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    setResult(data);
  };

  return (
    <div style={{ textAlign: "center", padding: "40px" }}>
      <h1>🌱 AgriAI</h1>

      <input
        type="file"
        accept="image/*"
        onChange={handleFileChange}
      />

      <br /><br />

      {preview && (
        <img
          src={preview}
          alt="Preview"
          width="300"
        />
      )}

      <br /><br />

      <button onClick={uploadImage}>
        Analyze Crop
      </button>

      {result && (
        <div style={{ marginTop: "20px" }}>
          <h2>Analysis Result</h2>

          <p><strong>Disease:</strong> {result.disease}</p>

          <p><strong>Confidence:</strong> {result.confidence}%</p>

          <p><strong>Health Score:</strong> {result.health_score}%</p>

          <p><strong>Recommendation:</strong></p>
          <p>{result.recommendation}</p>
        </div>
      )}
    </div>
  );
}

export default App;