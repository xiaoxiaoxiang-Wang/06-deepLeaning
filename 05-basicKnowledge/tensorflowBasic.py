import tensorflow as tf

w = tf.Variable([[1.0,2.0]])
with tf.GradientTape() as tape:
    loss = [w * w, w]

grad = tape.gradient(loss, w)
print(grad)
