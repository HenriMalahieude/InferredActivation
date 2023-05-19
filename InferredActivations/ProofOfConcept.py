import numpy as np
import tensorflow as tf
import keras
import time
from keras import datasets, layers, models, losses

(x_train,y_train),(x_test,y_test) = datasets.mnist.load_data()
x_train.shape

x_train = tf.pad(x_train, [[0, 0], [2,2], [2,2]])/255
x_test = tf.pad(x_test, [[0, 0], [2,2], [2,2]])/255
x_train.shape

#LesNet requires 32x32
x_train = tf.expand_dims(x_train, axis=3, name=None)
x_test = tf.expand_dims(x_test, axis=3, name=None)
x_train.shape

#Remove some data so we can validate
x_val = x_train[-2000:,:,:,:] 
y_val = y_train[-2000:] 
x_train = x_train[:-2000,:,:,:] 
y_train = y_train[:-2000]

print(x_train.shape[1:])

les_control = models.Sequential()
les_control.add(layers.Conv2D(6, 5, activation='tanh', input_shape=x_train.shape[1:]))
les_control.add(layers.AveragePooling2D(2))

les_control.add(layers.Activation('sigmoid'))
les_control.add(layers.Conv2D(16, 5, activation='tanh'))
les_control.add(layers.AveragePooling2D(2))

les_control.add(layers.Activation('sigmoid'))
les_control.add(layers.Conv2D(120, 5, activation='tanh'))

les_control.add(layers.Flatten())
les_control.add(layers.Dense(84, activation='tanh'))
les_control.add(layers.Dense(10, activation='softmax'))

print(les_control.summary())

les_control.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])
t_s = time.time()
c_history = les_control.fit(x_train, y_train, batch_size=64, epochs=5, validation_data=(x_val, y_val))
c_time = (time.time() - t_s) * 1000
print("Control Training Time: " + str(int(c_time)) + "ms")

#In all honesty, python has a horrible OOP implementation
class InferredSigmoid(layers.Layer):
      def __init__(self, randomize=False):
          super(InferredSigmoid, self).__init__()
          self.random_initialization = randomize
      def build(self, input_shape):
          pwlInit = ('random_normal' if self.random_initialization else 'one')
          self.pwlParams = self.add_weight(shape=(17,), initializer=pwlInit, trainable=True)

          if (self.random_initialization == False):
            #Boundary Conditions
            self.set_weights(np.array([[-6, -3.4, 0, 3.4, 6, 
                                        0, 
                                        0.011, 0.071, 
                                        0.75, 2, 0.5, 
                                        0.75, 2, 0.5, 
                                        0.011, 0.929, 
                                        1]]))

      def call(self, inputs):
          #Start the Piece-Wise Linear thingy (let us commence forth)
          #Calculate Boundaries... With this much math, it makes me wonder if this will really increase efficiency (compared to a single calc for all of the tensor by sigmoid)
          b1 = tf.cast(tf.math.less_equal(inputs, self.pwlParams[0]), inputs.dtype)
          b2 = tf.cast(tf.math.logical_and(tf.math.greater(inputs, self.pwlParams[0]), tf.math.less_equal(inputs, self.pwlParams[1])), inputs.dtype)
          b3 = tf.cast(tf.math.logical_and(tf.math.greater(inputs, self.pwlParams[1]), tf.math.less_equal(inputs, self.pwlParams[2])), inputs.dtype)
          b4 = tf.cast(tf.math.logical_and(tf.math.greater(inputs, self.pwlParams[2]), tf.math.less_equal(inputs, self.pwlParams[3])), inputs.dtype)
          b5 = tf.cast(tf.math.logical_and(tf.math.greater(inputs, self.pwlParams[3]), tf.math.less_equal(inputs, self.pwlParams[4])), inputs.dtype)
          b6 = tf.cast(tf.math.greater(inputs, self.pwlParams[4]), inputs.dtype)

          #Calculate Each Linear Piece
          l1 = tf.math.multiply(b1, self.pwlParams[5])
          l2 = tf.math.multiply(b2, tf.math.add(tf.math.multiply(inputs, self.pwlParams[6]), self.pwlParams[7]))
          l3 = tf.math.multiply(b3, tf.math.add(tf.math.divide_no_nan(tf.math.multiply(inputs, self.pwlParams[8]), tf.math.subtract(inputs, self.pwlParams[9])), self.pwlParams[10]))
          l4 = tf.math.multiply(b4, tf.math.add(tf.math.divide_no_nan(tf.math.multiply(inputs, self.pwlParams[11]), tf.math.add(inputs, self.pwlParams[12])), self.pwlParams[13]))
          l5 = tf.math.multiply(b5, tf.math.add(tf.math.multiply(inputs, self.pwlParams[14]), self.pwlParams[15]))
          l6 = tf.math.multiply(b6, self.pwlParams[16])

          return l1 + l2 + l3 + l4 + l5 + l6

custom_sigmoid = models.Sequential()
custom_sigmoid.add(layers.Conv2D(6, 5, activation='tanh', input_shape=x_train.shape[1:]))
custom_sigmoid.add(layers.AveragePooling2D(2))

custom_sigmoid.add(InferredSigmoid())
#print(custom_sigmoid.layers[2].get_weights());

custom_sigmoid.add(layers.Conv2D(16, 5, activation='tanh'))
custom_sigmoid.add(layers.AveragePooling2D(2))

custom_sigmoid.add(InferredSigmoid())
custom_sigmoid.add(layers.Conv2D(120, 5, activation='tanh'))

custom_sigmoid.add(layers.Flatten())
custom_sigmoid.add(layers.Dense(84, activation='tanh'))
custom_sigmoid.add(layers.Dense(10, activation='softmax'))

print(custom_sigmoid.summary(line_length=150))
custom_sigmoid.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])
t_s = time.time()
is_history = custom_sigmoid.fit(x_train, y_train, batch_size=64, epochs=5, validation_data=(x_val, y_val))
is_time = (time.time() - t_s) * 1000
print("Inferred Sigmoid Training Time: " + str(int(is_time)) + "ms")

#NOTE: This does need to get rather lucky due to the random nature
custom_rand_sigmoid = models.Sequential()
custom_rand_sigmoid.add(layers.Conv2D(6, 5, activation='tanh', input_shape=x_train.shape[1:]))
custom_rand_sigmoid.add(layers.AveragePooling2D(2))

custom_rand_sigmoid.add(InferredSigmoid(randomize=True))
#print(custom_rand_sigmoid.layers[2].get_weights());

custom_rand_sigmoid.add(layers.Conv2D(16, 5, activation='tanh'))
custom_rand_sigmoid.add(layers.AveragePooling2D(2))

custom_rand_sigmoid.add(InferredSigmoid(randomize=True))
custom_rand_sigmoid.add(layers.Conv2D(120, 5, activation='tanh'))

custom_rand_sigmoid.add(layers.Flatten())
custom_rand_sigmoid.add(layers.Dense(84, activation='tanh'))
custom_rand_sigmoid.add(layers.Dense(10, activation='softmax'))

print(custom_rand_sigmoid.summary(line_length=150))
custom_rand_sigmoid.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])
t_s = time.time()
irs_history = custom_rand_sigmoid.fit(x_train, y_train, batch_size=64, epochs=5, validation_data=(x_val, y_val))
irs_time = (time.time() - t_s) * 1000
print("Inferred Randomized Sigmoid Training Time: " + str(int(irs_time)) + "ms")

print("Final Accuracies: ", c_history.history['val_accuracy'], is_history.history['val_accuracy'], irs_history.history['val_accuracy'])
