import React from "react";

const PredictionResult = ({ prediction }) => {
  if (!prediction) return null;

  const predictedChar = prediction.prediction || "?";
  const confidenceValue = parseFloat(prediction.confidence);
  const confidencePercent = !isNaN(confidenceValue)
    ? (confidenceValue * 100).toFixed(2)
    : "0.00";

  return (
    <div className="mt-4 text-center">
      <h2 className="text-2xl font-bold text-green-600">
        Predicted Character: {predictedChar}
      </h2>
      <p className="text-gray-700 mt-1 text-lg">
        Confidence: <span className="font-medium">{confidencePercent}%</span>
      </p>
    </div>
  );
};

export default PredictionResult;