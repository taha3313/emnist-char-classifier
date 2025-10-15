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

model = tf.keras.models.load_model("model/mnist_cnn.h5")

def preprocess_image(file_bytes, is_raw=False):
    if is_raw:
        # Raw MNIST array bytes (shape 28x28)
        img_array = np.frombuffer(file_bytes, dtype=np.uint8).reshape((28,28)).astype("float32") / 255.0
        img_array = img_array.reshape((1,28,28,1))
        return img_array
    else:
        # Image files (PNG/BMP/JPEG)
        image = Image.open(io.BytesIO(file_bytes)).convert("L")
        if image.size != (28,28):
            image = image.resize((28,28))
        img_array = np.array(image).astype("float32") / 255.0
        img_array = img_array.reshape((1,28,28,1))
        return img_array

@app.post("/predict")
async def predict_digit(file: UploadFile = File(...), raw: bool = False):
    # raw=True means we expect raw MNIST array bytes
    image_data = await file.read()
    preprocessed = preprocess_image(image_data, is_raw=raw)
    
    predictions = model.predict(preprocessed)
    predicted_digit = int(np.argmax(predictions))
    confidence = float(np.max(predictions))
    return {"prediction": predicted_digit, "confidence": confidence}


@app.post("/predict_raw")
async def predict_raw(request: Request):
    body = await request.body()
    img_array = np.frombuffer(body, dtype=np.uint8).reshape((28,28)).astype("float32") / 255.0
    img_array = img_array.reshape((1,28,28,1))
    predictions = model.predict(img_array)
    return {"prediction": int(np.argmax(predictions)), "confidence": float(np.max(predictions))}