import React, { useState } from "react";
import { predictDigit } from "../services/api";
import PredictionResult from "./PredictionResult";

const ImageUploader = () => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
    setPrediction(null);
  };

  const handlePredict = async () => {
    if (!file) return;
    setLoading(true);
    try {
      const result = await predictDigit(file);
      setPrediction(result.prediction);
    } catch (error) {
      console.error("Prediction failed:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center">
      <input
        type="file"
        accept="image/*"
        onChange={handleFileChange}
        className="mb-4"
      />

      {preview && (
        <img
          src={preview}
          alt="preview"
          className="w-28 h-28 border-2 border-gray-400 mb-4 rounded-lg shadow-md"
        />
      )}

      <button
        onClick={handlePredict}
        disabled={!file || loading}
        className="bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600 disabled:bg-gray-400"
      >
        {loading ? "Predicting..." : "Predict Digit"}
      </button>

      <PredictionResult prediction={prediction} />
    </div>
  );
};

export default ImageUploader;