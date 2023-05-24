import tensorflow as tf
import numpy as np
from keras import layers

class PiecewiseLinearUnitV1(layers.Layer):
	def __init__(self, max_params=20):
		super(PiecewiseLinearUnitV1, self).__init__()
		self.max_n = max(max_params, 10) #We want to ensure at least a minium

	def build(self, input_shape):
		#TODO: Statistical Analysis
		self.N = self.add_weights(shape=(1,), initializer='one', trainable=True)
		self.Bounds = self.add_weights(shape=(2,), initializer='one', trainable=True)
		self.BoundSlope = self.add_weights(shape=(2,), initializer='one', trainable=True)
		self.nheight = self.add_weights(shape=(self.max_n+1,), initializer='one', trainable=True)
		self.set_weights([np.array([2]), np.array([-1, 1]), np.array([0, 0]), np.random.random_sample(size=(self.max_n+1,))])
	
	def call(self, inputs):
		intervals = max(int(self.N), self.max_n)
		Br, Bl = self.Bounds[1], self.Bounds[0]
		Kr, Kl = self.BoundSlope[1], self.BoundSlope[0]

		interval_length = (Br - Bl) / intervals
		
		Bidx_tensor = tf.math.floor(inputs)
		#Oh my fucking god I wasn't ready for the bs from tensorflow
		#NOTE: tf.gather_nd(indices=[[[0], [1]], [[2], [3]]], params=['a', 'b', 'c', 'd']) --> [['a', 'b'], ['c', 'd']]
		#So basically, we have a way to recontruct our.... tensors....?
		
		#This is the index tensor which will be indexing the self.nheight params...
		idx_tensor = tf.math.floor(tf.math.divide_no_nan(inputs - Bl, interval_length))
		idx_tensor = tf.map_fn(fn=lambda x: tf.range(x, x+1), elems=idx_tensor) #[[1, 5, 8], [., ., .]] ---> [[[1], [5], [8]], [[.],[.],[.]], etc]

		#And here we do the indexing of the params into a new tensor, which is now the nheights at each location...
		Yidx1_tensor = tf.gather_nd(params=self.nheight, indices=tf.math.add(idx_tensor, 1))
		Yidx0_tensor = tf.gather_nd(params=self.nheight, indices=idx_tensor)

		#And then we finally get what we need. . .
		Kidx_tensor = tf.math.divide_no_nan(tf.subtract(Yidx1_tensor, Yidx0_tensor), interval_length)

		#And now we can FINALLY get to the piece_wise_linear
		b1 = tf.cast(tf.math.less(inputs, Bl), dtype=inputs.dtype)
		b2 = tf.cast(tf.math.greater_equal(inputs, Br), dtype=inputs.dtype)
		b3 = tf.cast(tf.math.logical_and(tf.math.greater_equal(inputs, Bl), tf.math.less(inputs, Br)), dtype=inputs.dtype)

		l1 = (inputs - Bl) * Kl + self.nheight[0]
		l2 = (inputs - Br) * Kr + self.nheight[intervals+1]
		l3 = (inputs - Bidx_tensor) * Kidx_tensor + Yidx0_tensor

		#Holy shit this was so fucking confusing....
		return b1*l1 + b2*l2 + b3*l3