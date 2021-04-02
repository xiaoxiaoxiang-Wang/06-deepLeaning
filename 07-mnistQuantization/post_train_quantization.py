import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow import keras
from tensorflow.python.keras import losses

# Load MNIST dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the input image so that each pixel value is between 0 to 1.
train_images = train_images / 255.0
train_images = train_images[...,np.newaxis]
test_images = test_images / 255.0
test_images = test_images[...,np.newaxis]
print(test_images.shape)
# Define the model architecture.
model = keras.Sequential([
    # keras.layers.InputLayer(input_shape=(28, 28)),
    # keras.layers.Reshape(target_shape=(28, 28, 1)),
    keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu',input_shape=(28,28,1)),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(10)
])

# Train the digit classification model

model.compile(optimizer='adam',
              loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(
    train_images,
    train_labels,
    epochs=1,
    validation_split=0.1,
)

model.save("./model.h5")

# converter = tf.lite.TFLiteConverter.from_keras_model(model)
#
# tflite_model = converter.convert()
#
# print("model",tflite_model)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model_quant = converter.convert()

print("model",tflite_model_quant)