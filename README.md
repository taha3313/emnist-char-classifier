# EMNIST Handwritten Character Classifier — Deep Learning + FastAPI + React

A **handwritten character recognition** web application built with:

- **TensorFlow** (CNN trained on EMNIST)
- **FastAPI** backend
- **React 18 + Vite + Tailwind 4** frontend

The app allows users to upload images and get real-time character predictions.

The repository also includes **train_model.py**, the script I wrote to train the CNN model on the EMNIST dataset.

---

## 🛠 Features

- CNN model trained on EMNIST dataset (digits + letters)
- FastAPI backend serving predictions
- React 18 frontend with Tailwind 4 styling
- Image upload with **preview** before prediction
- Returns **predicted character** and **confidence score**
- CORS enabled for easy frontend-backend communication

---

## 📁 Project Structure

```
mnist-digit-classifier/
├─ backend/
│  ├─ app.py             # FastAPI backend
│  ├─ model/
│  │  └─ emnist_cnn.h5   # Pre-trained CNN model
│  ├─ train_model.py      # Script to train the CNN model
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
tensorflow_datasets
```

### 3. Run the backend

```bash
uvicorn app:app --reload
```

- Runs at `http://127.0.0.1:8000`
- `/predict` endpoint accepts uploaded images

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
  - `PredictionResult.jsx` → shows predicted character and confidence
- Tailwind 4 provides styling via **Vite configuration** (`vite.config.js`)

---

## 🔹 API Endpoints

### POST `/predict`

- Upload image (`PNG`, `JPEG`, `BMP`)

**Example using Python:**

```python
import requests

url = "http://127.0.0.1:8000/predict"
files = {"file": open("digit_or_letter.png", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

**Response:**

```json
{
  "prediction": "A",
  "confidence": 0.9987
}
```

---

## ⚠️ Notes

- Model is trained on **EMNIST** (digits + letters); real-world images may require **data augmentation**
- Backend preprocessing handles resizing, normalization, rotation, and optional color inversion

---

## 📚 References

- [EMNIST Dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset)
- [TensorFlow CNN Tutorial](https://www.tensorflow.org/tutorials/quickstart/beginner)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React 18](https://reactjs.org/)
- [Vite](https://vitejs.dev/)
- [Tailwind 4](https://tailwindcss.com/)