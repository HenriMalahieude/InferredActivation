import tensorflow as tf
import numpy as np
from keras import layers

class PiecewiseLinearUnitV1(layers.Layer):
	def __init__(self, max_params=20, interval_start=3):
		super(PiecewiseLinearUnitV1, self).__init__()
		self.max_n = max(max_params, 10) #We want to ensure at least a minium of parameters
		self.N_start = max(min(int(interval_start), 20), 2)

	def build(self, input_shape):
		#TODO: Statistical Analysis of input to then give a best initial status
		self.N = self.add_weight(shape=(), initializer='one', trainable=True)
		self.Bounds = self.add_weight(shape=(2,), initializer='one', trainable=True)
		self.BoundSlope = self.add_weight(shape=(2,), initializer='one', trainable=True)
		self.nheight = self.add_weight(shape=(self.max_n+1,), initializer='one', trainable=True)
		self.set_weights([self.N_start, np.array([-1, 1]), np.array([0, 0]), np.random.random_sample(size=(self.max_n+1,))])
	
	def call(self, inputs):

		@tf.custom_gradient #Let's test this custom gradient thing
		def intervalFunc(cur):
			def grad(x):
				return x*cur 
			
			return tf.math.floor(tf.math.maximum(tf.math.minimum(cur, self.max_n), 2)), grad

		intervals = intervalFunc(self.N)
		Br, Bl = self.Bounds[1], self.Bounds[0]
		Kr, Kl = self.BoundSlope[1], self.BoundSlope[0]

		interval_length = (Br - Bl) / intervals
		
		#TODO: There is one thing left, that is making self.nheights[self.N+2] = self.nheights[self.N+1] if self.N != self.max_n
		#		-> the reason for this is that when it reaches the next "interval" size, the right bound height suddenly becomes completely random making it think it 
		# 			shouldn't grow the parameters if it's not a good height by chance

		#This is the index tensor which will be indexing the self.nheight params...
		idx_tensor = tf.math.floor(tf.math.divide_no_nan(inputs - Bl, interval_length))
		Bidx_tensor = idx_tensor * interval_length + Bl

		#Oh my fucking god I wasn't ready for the bs from tensorflow
		#NOTE: tf.gather_nd(indices=[[[0], [1]], [[2], [3]]], params=['a', 'b', 'c', 'd']) --> [['a', 'b'], ['c', 'd']]
		#So basically, we have a way to recontruct our.... tensors....?
		idx_tensor = tf.expand_dims(idx_tensor, axis=len(idx_tensor.shape.as_list()))
		idx_tensor = tf.cast(idx_tensor, dtype=tf.int32)

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
		l2 = (inputs - Br) * Kr + tf.gather(self.nheight, tf.cast(intervals+1, dtype=tf.int32))
		l3 = (inputs - Bidx_tensor) * Kidx_tensor + Yidx0_tensor

		#Holy shit this was so fucking confusing....
		return b1*l1 + b2*l2 + b3*l3
	
	def Extract(self):
		print("Boundary Info:\n", 
			"Bl = " + str(self.Bounds[0].numpy()), "w/slope=", str(self.BoundSlope[0].numpy()), "\n",
			"Br = " + str(self.Bounds[1].numpy()), "w/slope=", str(self.BoundSlope[1].numpy()), "\n")
		
		print("All Heights (Interval Count = " + str(self.N.numpy()) + "): ")
		m = " "
		for i in range(int(self.N.numpy()+1)):
			m = m + str(self.nheight[i].numpy()) + ", "
		
		print(m)