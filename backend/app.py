import io
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained EMNIST model
model = tf.keras.models.load_model("model/emnist_cnn.h5")

# Mapping for EMNIST "byclass" labels (0–61)
# Reference: https://www.nist.gov/itl/products-and-services/emnist-dataset
emnist_labels = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',  # digits
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z',                      # uppercase letters
    'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r',
    't', 'y'                                           # lowercase subset
]

def preprocess_image(file_bytes, is_raw=False):
    """Convert uploaded file or raw bytes into normalized 28x28 grayscale tensor"""
    if is_raw:
        img_array = np.frombuffer(file_bytes, dtype=np.uint8).reshape((28, 28)).astype("float32") / 255.0
    else:
        image = Image.open(io.BytesIO(file_bytes)).convert("L")
        if image.size != (28, 28):
            image = image.resize((28, 28))
        img_array = np.array(image).astype("float32") / 255.0
    
    # EMNIST images are rotated 90° and flipped — fix orientation
    img_array = np.rot90(img_array, k=-1)
    img_array = np.fliplr(img_array)
    
    return img_array.reshape((1, 28, 28, 1))


@app.post("/predict")
async def predict_character(file: UploadFile = File(...), raw: bool = False):
    """Predict the handwritten digit or letter from uploaded image"""
    image_data = await file.read()
    preprocessed = preprocess_image(image_data, is_raw=raw)

    predictions = model.predict(preprocessed)
    pred_idx = int(np.argmax(predictions))
    confidence = float(np.max(predictions))

    # Map numeric label to actual character
    predicted_char = emnist_labels[pred_idx] if pred_idx < len(emnist_labels) else "?"

    return {"prediction": predicted_char, "index": pred_idx, "confidence": confidence}