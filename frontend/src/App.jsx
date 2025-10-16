import React from "react";
import ImageUploader from "./components/ImageUploader";

const App = () => {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-100 p-6">
      <h1 className="text-3xl font-bold mb-6 text-gray-800">
        🧠 EMNIST Character Classifier
      </h1>
      <ImageUploader />
    </div>
  );
};

export default App;