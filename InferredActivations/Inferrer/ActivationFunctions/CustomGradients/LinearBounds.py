import tensorflow as tf

#NOTE: This requires that self have "bounds" and "pwlParams" 
def OuterBound(self, input, paramIndex = [0, 0], boundIndex = 0, top = False):
	assert len(paramIndex) == 2

	@tf.custom_gradient
	def Internal(x):
		bound = tf.cast(tf.math.less_equal(x, self.bounds[boundIndex]), dtype=x.dtype)
		if top:
			bound = tf.cast(tf.math.greater(x, self.bounds[boundIndex]),  dtype=x.dtype)

		y = bound * (x * self.pwlParams[paramIndex[0]] + self.pwlParams[paramIndex[1]])
		def grad_fn(dx, variables): #Note that variables argument is keyword required
			dy_dx = dx * self.pwlParams[0]
			dscalar = tf.reshape(tf.reduce_mean(bound * dx), [1])
			dbound = tf.pad(dscalar, [[boundIndex, variables[0].shape.as_list()[0] - (boundIndex + 1)]])
			dparams = tf.pad(dscalar, [[paramIndex[0], variables[1].shape.as_list()[0] - (paramIndex[0] + 1)]])
			#print("Out EQ:", dy_dx, grad_vars[0], grad_vars[1], sep="\n\t")
			return dy_dx, [dbound, dparams]
		return y, grad_fn
	
	return Internal(input)

#NOTE: Expects only size 2 parameter and bound indices
def InnerBound(self, input, paramIndex = [1, 2], boundIndex = [0, 1]):
	assert len(paramIndex) == 2
	assert len(boundIndex) == 2

	@tf.custom_gradient
	def Internal(x):
		bound = tf.cast(tf.math.logical_and(tf.math.greater(x, self.bounds[boundIndex[0]]), tf.math.less_equal(x, self.bounds[boundIndex[1]])), dtype=x.dtype)
		y = bound * (x * self.pwlParams[paramIndex[0]] + self.pwlParams[paramIndex[1]])
		def grad_fn(dx, variables):
			dy_dx = dx * self.pwlParams[paramIndex[0]]
			dscalar = tf.constant([1, 1], dtype=x.dtype) * tf.reduce_mean(bound * dx)
			dbounds = tf.pad(dscalar, [[boundIndex[0], variables[0].shape.as_list()[0] - (boundIndex[0] - 2)]])
			dparams = tf.pad(dscalar, [[paramIndex[0], variables[1].shape.as_list()[0] - (paramIndex[0] - 2)]])
			#print("In Eq:", dy_dx, dbounds, dparams, sep="\n\t")
			return dy_dx, [dbounds, dparams]
		return y, grad_fn

	return Internal(input)
