import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import mnist

# Import EMNIST from tensorflow_datasets
import tensorflow_datasets as tfds

# Load the EMNIST "byclass" dataset (digits + letters)
(ds_train, ds_test), ds_info = tfds.load(
    'emnist/byclass',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

# Preprocess function: normalize images and reshape
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0  # Normalize
    image = tf.expand_dims(image, -1)           # Add channel dimension (28,28,1)
    return image, label

# Apply preprocessing
ds_train = ds_train.map(preprocess).batch(64).prefetch(tf.data.AUTOTUNE)
ds_test = ds_test.map(preprocess).batch(64).prefetch(tf.data.AUTOTUNE)

# Get number of classes (should be 62 for EMNIST ByClass)
num_classes = ds_info.features['label'].num_classes
print(f"Number of classes: {num_classes}")

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')  # 62 classes for EMNIST
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(ds_train, epochs=5, validation_data=ds_test)

# Evaluate model
test_loss, test_acc = model.evaluate(ds_test)
print(f'Test accuracy: {test_acc * 100:.2f}%')

# Predict example
for images, labels in ds_test.take(1):
    preds = model.predict(images)
    plt.imshow(images[0].numpy().reshape(28, 28), cmap='gray')
    plt.title(f"Predicted Label: {np.argmax(preds[0])}")
    plt.show()

# Save model
model.save("model/emnist_cnn.h5")
print("✅ Model saved to model/emnist_cnn.h5")