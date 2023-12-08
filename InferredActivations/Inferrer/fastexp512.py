import tensorflow as tf
from keras import layers

class fastexp512_act(layers.Layer):
	def __init__(self, n_init:int = 8, n_locked:bool = True):
		self.n_init = n_init
		self.locked = n_locked
		super(fastexp512_act, self).__init__()

	def build(self, input_shape):
		#self.n = self.add_weight(shape=(), initializer='one', trainable=(~self.locked), name="n")
		#self.set_weights([self.n_init])
		return

	def call(self, input):
		"""@tf.custom_gradient
		def thing(cur):
			y = tf.math.floor(cur)

			def grad(up):
				return up * cur
			
			return y, grad"""
		
		minus_x = -1 * input
		exp = tf.math.pow((1 + minus_x), 0)

		for i in range(1, self.n_init):
			ii = tf.cast(i, dtype=minus_x.dtype)
			exp = exp * tf.math.pow((1 + tf.math.divide_no_nan(minus_x, ii)), ii)

		return tf.math.divide_no_nan(1, (1 + exp))