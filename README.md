# MNIST Digit Classifier

A **handwritten digit recognition** web application built with:

- **TensorFlow** (CNN trained on MNIST)  
- **FastAPI** backend  
- **React 18 + Vite + Tailwind 4** frontend  

The app allows users to upload images and get real-time digit predictions.

---

## 🛠 Features

- CNN model trained on MNIST dataset  
- FastAPI backend serving predictions  
- React 18 frontend with Tailwind 4 styling  
- Image upload with **preview** before prediction  
- Returns **predicted digit** and **confidence score**  
- CORS enabled for easy frontend-backend communication  

---

## 📁 Project Structure

```
mnist-digit-classifier/
├─ backend/
│  ├─ app.py             # FastAPI backend
│  ├─ model/
│  │  └─ mnist_cnn.h5   # Pre-trained CNN model
│  └─ venv/              # Python virtual environment
├─ frontend/
│  ├─ src/
│  │  ├─ components/
│  │  │  ├─ ImageUploader.jsx
│  │  │  └─ PredictionResult.jsx
│  │  ├─ services/
│  │  │  └─ api.js       # Axios requests to backend
│  │  └─ App.jsx
│  ├─ package.json
│  └─ vite.config.js     # Vite + Tailwind 4 configuration
└─ README.md
```

---

## ⚡ Backend Setup

### 1. Create a virtual environment

```bash
cd backend
python -m venv venv
venv\Scripts\activate      # Windows
# OR source venv/bin/activate # Linux / macOS
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt**:

```
tensorflow
numpy
matplotlib
fastapi
uvicorn
pillow
python-multipart
opencv-python
```

### 3. Run the backend

```bash
uvicorn app:app --reload
```

- Runs at `http://127.0.0.1:8000`  
- `/predict` endpoint accepts uploaded images  
- `/predict_raw` endpoint accepts raw MNIST byte arrays  

---

## ⚡ Frontend Setup (React 18 + Vite + Tailwind 4)

### 1. Install dependencies

```bash
cd frontend
npm install
```

### 2. Run development server

```bash
npm run dev
```

- Runs at `http://localhost:5173` (Vite default)  
- Upload images to get predictions from the backend  

### 3. Notes

- Frontend is **component-based**:
  - `ImageUploader.jsx` → handles file upload & prediction  
  - `PredictionResult.jsx` → shows predicted digit  
- Tailwind 4 provides styling via **Vite configuration** (`vite.config.js`)  

---

## 🔹 API Endpoints

### POST `/predict`

- Upload image (`PNG`, `JPEG`, `BMP`)  
- Optional query param: `raw=true` for raw MNIST arrays  

**Example using Python:**

```python
import requests

url = "http://127.0.0.1:8000/predict"
files = {"file": open("digit.png", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

**Response:**

```json
{
  "prediction": 7,
  "confidence": 0.9987
}
```

---

### POST `/predict_raw`

- Accepts raw MNIST 28x28 array bytes in the request body  

```python
import requests
import numpy as np

url = "http://127.0.0.1:8000/predict_raw"
mnist_sample = np.random.randint(0, 255, (28,28), dtype=np.uint8)
response = requests.post(url, data=mnist_sample.tobytes())
print(response.json())
```

---

## ⚠️ Notes

- Model is trained only on MNIST; real-world images may require **data augmentation** or **EMNIST**  
- Backend preprocessing handles resizing, normalization, and optional color inversion  

---

## 📚 References

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)  
- [TensorFlow CNN Tutorial](https://www.tensorflow.org/tutorials/quickstart/beginner)  
- [FastAPI Documentation](https://fastapi.tiangolo.com/)  
- [React 18](https://reactjs.org/)  
- [Vite](https://vitejs.dev/)  
- [Tailwind 4](https://tailwi