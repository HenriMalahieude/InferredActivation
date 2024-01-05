import tensorflow as tf
import numpy as np
from keras import layers

class ShiftReLU(layers.Layer):
	def __init__(self, start=0):
		super(ShiftReLU, self).__init__()
		self.shift = self.add_weight(shape=(), initializer="one", trainable=True)
		self.set_weights([start])

	def call(self, inputs):
		x = inputs + self.shift
		return tf.nn.relu(x)
	
class LeakyShiftReLU(layers.Layer):
	def __init__(self, start_shift=0, start_leak=0):
		super(LeakyShiftReLU, self).__init__()
		self.shift = self.add_weight(shape=(), initializer="one", trainable=True)
		self.leak = self.add_weight(shape=(), intializer="one", trainable=True)

		self.set_weights([start_shift, start_leak])

	def call(self, inputs):
		x = inputs + self.shift

		return (tf.math.min(x, 0) * self.leak) + tf.nn.relu(x)