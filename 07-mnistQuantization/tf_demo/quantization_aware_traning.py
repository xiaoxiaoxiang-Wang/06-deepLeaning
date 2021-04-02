import tempfile

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow import keras

# Load MNIST dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the input image so that each pixel value is between 0 to 1.
train_images = train_images / 255.0
test_images = test_images / 255.0

# Define the model architecture.
model = keras.Sequential([
    keras.layers.InputLayer(batch_size=1, input_shape=(28, 28)),
    keras.layers.Reshape(target_shape=(28, 28, 1)),
    keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(10)
])

print("shape=", model.output_shape)

# Train the digit classification model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(
    train_images,
    train_labels,
    batch_size=1,
    epochs=1,
    validation_split=0.1,
)

# q_aware stands for for quantization aware.
q_aware_model = tfmot.quantization.keras.quantize_model(model)

# `quantize_model` requires a recompile.
q_aware_model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

train_images_subset = train_images  # out of 60000
train_labels_subset = train_labels

q_aware_model.fit(train_images_subset, train_labels_subset,
                  batch_size=500, epochs=1, validation_split=0.1)
q_aware_model.summary()

# _, baseline_model_accuracy = model.evaluate(
#     test_images, test_labels, verbose=0)
#
# _, q_aware_model_accuracy = q_aware_model.evaluate(
#     test_images, test_labels, verbose=0)
#
# # print("model", model.get_weights())
# # print("quantize_model", q_aware_model.get_weights())
#
# print('Baseline test accuracy:', baseline_model_accuracy)
# print('Quant test accuracy:', q_aware_model_accuracy)
#
converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

quantized_tflite_model = converter.convert()
#
#
# # open("../converted_model.tflite", "wb").write(tflite_model)
# #
# # freeze_graph
#
# def evaluate_model(interpreter):
#     input_index = interpreter.get_input_details()[0]["index"]
#     output_index = interpreter.get_output_details()[0]["index"]
#
#     # Run predictions on every image in the "test" dataset.
#     prediction_digits = []
#     for i, test_image in enumerate(test_images):
#         if i % 1000 == 0:
#             print('Evaluated on {n} results so far.'.format(n=i))
#         # Pre-processing: add batch dimension and convert to float32 to match with
#         # the model's input data format.
#         test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
#         interpreter.set_tensor(input_index, test_image)
#
#         # Run inference.
#         interpreter.invoke()
#
#         # Post-processing: remove batch dimension and find the digit with highest
#         # probability.
#         output = interpreter.tensor(output_index)
#         digit = np.argmax(output()[0])
#         prediction_digits.append(digit)
#
#     print('\n')
#     # Compare prediction results with ground truth labels to calculate accuracy.
#     prediction_digits = np.array(prediction_digits)
#     accuracy = (prediction_digits == test_labels).mean()
#     return accuracy
#
# interpreter = tf.lite.Interpreter(model_content=quantized_tflite_model)
# interpreter.allocate_tensors()
#
# test_accuracy = evaluate_model(interpreter)
#
# print('Quant TFLite test_accuracy:', test_accuracy)
# print('Quant TF test accuracy:', q_aware_model_accuracy)

float_converter = tf.lite.TFLiteConverter.from_keras_model(model)
float_tflite_model = float_converter.convert()


with open('quant_file.tflite', 'wb') as f:
    f.write(quantized_tflite_model)

with open('float_file.tflite', 'wb') as f:
    f.write(float_tflite_model)
