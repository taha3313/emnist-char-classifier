# test.py
import requests
from tensorflow.keras.datasets import mnist
import numpy as np

(_, _), (test_images, test_labels) = mnist.load_data()
test_images_uint8 = (test_images * 255).astype(np.uint8)

url = "http://127.0.0.1:8000/predict_raw"
correct = 0
total = 100

for i in range(total):
    response = requests.post(url, data=test_images_uint8[i].tobytes())
    prediction = response.json()["prediction"]
    if prediction == int(test_labels[i]):
        correct += 1

accuracy = correct / total
print(f"Accuracy: {accuracy:.4f}")