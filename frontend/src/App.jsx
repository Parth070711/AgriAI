import { useState } from "react";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];

    setFile(selected);

    if (selected) {
      setPreview(URL.createObjectURL(selected));
    }
  };

  const uploadImage = async () => {
    if (!file) {
      alert("Please select an image.");
      return;
    }

    setLoading(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(
        "https://agriai-backend-ctmq.onrender.com/",
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();

      setResult(data);
    } catch (error) {
      console.error(error);
      alert("Failed to connect to backend.");
    }

    setLoading(false);
  };

  return (
    <div
      style={{
        textAlign: "center",
        padding: "40px",
        fontFamily: "Arial",
      }}
    >
      <h1>🌱 AgriAI Crop Disease Detector</h1>

      <input
        type="file"
        accept="image/*"
        onChange={handleFileChange}
      />

      <br />
      <br />

      {preview && (
        <img
          src={preview}
          alt="Preview"
          width="300"
          style={{
            borderRadius: "10px",
            border: "2px solid #ddd",
          }}
        />
      )}

      <br />
      <br />

      <button
        onClick={uploadImage}
        style={{
          padding: "10px 20px",
          fontSize: "16px",
          cursor: "pointer",
        }}
      >
        Analyze Crop
      </button>

      <br />
      <br />

      {loading && (
        <h3>🔄 Analyzing Image...</h3>
      )}

      {result && (
        <div
          style={{
            marginTop: "20px",
            border: "1px solid #ddd",
            borderRadius: "10px",
            padding: "20px",
            maxWidth: "700px",
            marginLeft: "auto",
            marginRight: "auto",
          }}
        >
          <h2>Analysis Result</h2>

          <h3>
            🌿 Disease: {result.disease}
          </h3>

          <h3>
            🎯 Confidence: {result.confidence}%
          </h3>

          <h3>
            🟡 Severity: {result.severity}
          </h3>

          <h3>
            ❤️ Health Score: {result.health_score}%
          </h3>

          <h3>
            💡 Recommendation
          </h3>

          <p>
            {result.recommendation}
          </p>

          <h3>
            📖 About Disease
          </h3>

          <p>
            {result.about}
          </p>
        </div>
      )}
    </div>
  );
}

export default App;
