import numpy as np
import tensorflow as tf
import InferredActivations.Inferrer.ActivationLinearizer as AL
from keras import models

#"""
lyr = AL.ActivationLinearizer()
lyr.build(input_shape=(1,))

with tf.GradientTape(persistent = True) as tape:
    x = tf.constant([-7, -4, -2, 2, 4, 7], dtype = tf.float32)
    tape.watch(x)
    result = lyr.call(x)

print()
print("Input: ", x)
print("Result: ", result)
print("dy/dx: ", tape.gradient(result, x))
print("Gradients for pwl: ", tape.gradient(result, lyr.pwlParams))
print("Gradients for bounds: ", tape.gradient(result, lyr.bounds))
lyr.extract_linears()#"""

"""sigmoid_fn = tf.math.sigmoid

training_data_x = tf.constant(np.linspace(-15, 15, (2 ** 14)), dtype=tf.float32)
training_data_y = sigmoid_fn(training_data_x)

val_data_x = tf.constant((np.random.randn(2048) * 20 - 10), dtype=tf.float32)
val_data_y = sigmoid_fn(val_data_x)

linearizer = models.Sequential()
linearizer.add(AL.ActivationLinearizer(initial_eq='random'))
#linearizer.add(tf.keras.layers.ActivityRegularization(l1 = 0.01, l2 = 0.0))

linearizer.build(input_shape=(1,))

#linearizer.summary()

linearizer.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
linearizer.fit(x=training_data_x, y=training_data_y, epochs=5, validation_data=(val_data_x, val_data_y))

linearizer.layers[0].extract_linears()"""