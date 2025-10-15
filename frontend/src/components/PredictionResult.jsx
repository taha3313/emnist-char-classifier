import React from "react";

const PredictionResult = ({ prediction }) => {
  if (prediction === null) return null;

  return (
    <div className="mt-4 text-xl font-semibold text-green-600">
      Predicted Digit: {prediction}
    </div>
  );
};

export default PredictionResult;