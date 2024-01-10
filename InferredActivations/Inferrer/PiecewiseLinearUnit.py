import tensorflow as tf
import numpy as np
from keras import layers#, initializers

"""class StepperInitializer(initializers.Initializer):
	def __init__(self, bot, top, count):
		self.bot = bot
		self.step_size = (top - bot) / count
		self.iter = 0

	def __call__(self, shape, dtype=None, **kwargs):"""

def tensor_clamp(tensor, min, max):
	return tf.math.maximum(tf.math.minimum(tensor, max), min)

#OG Paper: https://arxiv.org/abs/2104.03693
#Referenced: https://github.com/MrGoriay/pwlu-pytorch/blob/main/PWLA.py
@tf.keras.saving.register_keras_serializable('InferredActivation')
class PiecewiseLinearUnitV1(layers.Layer):
	def __init__(self, k=5, momentum=0.9, left=None, right=None):
		assert (left is None and right is None) or (left < right)
		super(PiecewiseLinearUnitV1, self).__init__()

		self.K = k #NOTE: K is the amount of points, and not the amount of intervals
		self.ravg = 0.0 if left is None else ((right - left) / 2.0)
		self.rstd = (10.0/3) if left is None else ((right - self.ravg) / 3.0)
		self.momentum = momentum
		self.collect_stats = False

	def build(self, input_shape):
		self.Bounds = self.add_weight(shape=(2,), initializer='one', trainable=True)
		self.BoundSlope = self.add_weight(shape=(2,), initializer='one', trainable=True)
		self.nheight = self.add_weight(shape=(self.K,), initializer='one', trainable=True)

		l, r = self.ravg - (3*self.rstd), self.ravg + (3*self.rstd)
		thing = [np.array([l, r]), np.array([0, 1]), np.maximum(np.linspace(start=l, stop=r, num=self.K), 0)]
		self.set_weights(thing) #Defaults to ReLU init

	def call(self, inputs):
		#Has to be done during "call" of the layer or model. Can't be during evaluate() or fit(). Otherwise TensorFlow gets angry
		if self.collect_stats:
			self.ravg = (self.ravg * self.momentum) + (1 - self.momentum) * tf.math.reduce_mean(inputs)
			self.rstd = (self.rstd * self.momentum) + (1 - self.momentum) * tf.math.reduce_std(inputs)

			return tf.nn.relu(inputs)

		intervals = self.K - 1
		Br, Bl = self.Bounds[1], self.Bounds[0]
		Kr, Kl = self.BoundSlope[1], self.BoundSlope[0]

		interval_length = (Br - Bl) / intervals
	
		#This is the index tensor which will be indexing the self.nheight params...
		idx_tensor = tensor_clamp(tf.math.floor(tf.math.divide(inputs - Bl, interval_length)), 0, (self.K-1)) #can't have out of bounds
		Bidx_tensor = idx_tensor * interval_length + Bl

		#Oh my fucking god I wasn't ready for the bs from tensorflow
		#NOTE: tf.gather_nd(indices=[[[0], [1]], [[2], [3]]], params=['a', 'b', 'c', 'd']) --> [['a', 'b'], ['c', 'd']]
		#So basically, we have a way to recontruct our.... tensors....?
		idx_tensor = tf.cast(tf.expand_dims(idx_tensor, axis=len(idx_tensor.shape.as_list())), dtype=tf.int32)

		#And here we do the indexing of the params into a new tensor, which is now the nheights at each location...
		Yidx1_tensor = tf.gather_nd(params=self.nheight, indices=tf.math.add(idx_tensor, 1))
		Yidx0_tensor = tf.gather_nd(params=self.nheight, indices=idx_tensor)

		#And then we finally get what we need. . . (the heights)
		Kidx_tensor = tf.math.divide(tf.subtract(Yidx1_tensor, Yidx0_tensor), interval_length)

		#And now we can FINALLY get to the piece_wise_linear
		b1 = tf.cast(tf.math.less(inputs, Bl), dtype=inputs.dtype)
		b2 = tf.cast(tf.math.greater_equal(inputs, Br), dtype=inputs.dtype)
		b3 = tf.cast(tf.math.logical_and(tf.math.greater_equal(inputs, Bl), tf.math.less(inputs, Br)), dtype=inputs.dtype)

		l1 = (inputs - Bl) * Kl + self.nheight[0]
		l2 = (inputs - Br) * Kr + self.nheight[-1]
		l3 = (inputs - Bidx_tensor) * Kidx_tensor + Yidx0_tensor

		#Holy shit that was convoluted
		return b1*l1 + b2*l2 + b3*l3
	
	def Extract(self):
		print("Boundary Info:\n", 
			"Bl = " + str(self.Bounds[0].numpy()), "w/slope=", str(self.BoundSlope[0].numpy()), "\n",
			"Br = " + str(self.Bounds[1].numpy()), "w/slope=", str(self.BoundSlope[1].numpy()), "\n")
		
		print("All Heights (Interval Count = " + str(self.K) + "): ")
		m = " "
		for i in range(int(self.K + 1)):
			m = m + str(self.nheight[i].numpy()) + ", "
		
		print(m)
	
	def GetStatInfo(self):
		return self.ravg, self.rstd

	def StatisticalAnalysisToggle(self, forceTo=None):
		before = self.collect_stats
		if forceTo != None and type(forceTo) is bool:
			self.collect_stats = forceTo
		else:
			self.collect_stats = not self.collect_stats

		if before and not self.collect_stats: #Meaning we've ended our stats collection phase
			avg, std = self.GetStatInfo()
			dist = 3 * std

			Bl_stat = avg - dist
			Br_stat = avg + dist
			self.set_weights([np.array([Bl_stat, Br_stat]), np.array([0, 1]), tf.math.maximum(tf.linspace(start=tf.math.minimum(Bl_stat, -0.1), stop=Br_stat, num=self.K), 0)])
			#print(f"\nRunning Mean: {self.ravg}\n\tRunning StD: {self.rstd}")

#This one has parameter growth
class PiecewiseLinearUnitV2(layers.Layer):
	def __init__(self, max_params=20, interval_start=5, momentum=0.9):
		super(PiecewiseLinearUnitV2, self).__init__()
		self.max_n = max(max_params, 10)
		self.N_start = max(min(int(interval_start), 20), 3)
		
		self.running_avg = 0
		self.running_std = 1
		self.momentum = momentum
		self.boundary_lock = False

		self.collect_stats = False

	def get_config(self):
		base_config = super().get_config()
		config = {
			"max_n": self.max_n,
			"N_start": self.N_start,
			"running_avg": self.running_avg,
			"running_std": self.running_std,
			"momentum": self.momentum,
		}
		return {**base_config, **config}

	@classmethod
	def from_config(cls, config):
		lyr = PiecewiseLinearUnitV1(config["max_n"], config["N_start"], config["momentum"])
		lyr.running_avg = config["running_avg"]
		lyr.running_std = config["running_std"]
		lyr.collect_stats = False
		return lyr

	def build(self, input_shape):
		self.N = self.add_weight(shape=(), initializer='one', trainable=True)
		self.Bounds = self.add_weight(shape=(2,), initializer='one', trainable=True)
		self.BoundSlope = self.add_weight(shape=(2,), initializer='one', trainable=True)
		self.nheight = self.add_weight(shape=(self.max_n+1,), initializer='one', trainable=True)

		#Start as a ReLU, though there can be more options tried
		self.set_weights([self.N_start, np.array([-3, 3]), np.array([0, 1]), np.linspace(start=0, stop=6, num=self.max_n+1)]) #np.random.random_sample(size=(self.max_n+1,))
	
	def call(self, inputs):

		if self.collect_stats:
			avg = tf.math.reduce_mean(inputs)
			std = tf.math.reduce_std(inputs) #TODO: Rewrite this to avoid "eager execution"

			self.running_avg = (self.running_avg * self.momentum) + (1 - self.momentum) * avg.numpy()
			self.running_std = (self.running_std * self.momentum) + (1 - self.momentum) * std.numpy()

			#Default to ReLU while stats are collected
			b1 = tf.cast(tf.math.greater(inputs, 0.0), dtype=inputs.dtype)

			return b1 * inputs

		@tf.custom_gradient #Let's test this custom gradient thing
		def intervalFunc(cur):
			y = tf.math.floor(tf.math.maximum(tf.math.minimum(cur, self.max_n), 3))

			def grad(up):
				return up * cur * self.max_n #We want the current count to impact the gradient, and the maximum count to also force movement
			
			return y, grad

		intervals = intervalFunc(self.N) if not self.boundary_lock else tf.math.floor(self.N)
		Br, Bl = self.Bounds[1], self.Bounds[0]
		Kr, Kl = self.BoundSlope[1], self.BoundSlope[0]

		interval_length = (Br - Bl) / intervals
		
		#Proposal: There is one thing left, that is making self.nheights[self.N+2] = self.nheights[self.N+1] if self.N != self.max_n
		#		-> the reason for this is that when it reaches the next "interval" size, the right bound height suddenly becomes completely random making it think it 
		# 			shouldn't grow the parameters if it's not a good height by chance
		#NOTE: Fixed this by simply doing "linspace", though the network could train the other weights not used yet in hope of change which could be.... worrisome.

		#This is the index tensor which will be indexing the self.nheight params...
		idx_tensor = tf.math.floor(tf.math.divide(inputs - Bl, interval_length))
		Bidx_tensor = idx_tensor * interval_length + Bl

		#Oh my fucking god I wasn't ready for the bs from tensorflow
		#NOTE: tf.gather_nd(indices=[[[0], [1]], [[2], [3]]], params=['a', 'b', 'c', 'd']) --> [['a', 'b'], ['c', 'd']]
		#So basically, we have a way to recontruct our.... tensors....?
		idx_tensor = tf.cast(tf.expand_dims(idx_tensor, axis=len(idx_tensor.shape.as_list())), dtype=tf.int32)

		#And here we do the indexing of the params into a new tensor, which is now the nheights at each location...
		Yidx1_tensor = tf.gather_nd(params=self.nheight, indices=tf.math.add(idx_tensor, 1))
		Yidx0_tensor = tf.gather_nd(params=self.nheight, indices=idx_tensor)

		#And then we finally get what we need. . . (the heights)
		Kidx_tensor = tf.math.divide(tf.subtract(Yidx1_tensor, Yidx0_tensor), interval_length)

		#And now we can FINALLY get to the piece_wise_linear
		b1 = tf.cast(tf.math.less(inputs, Bl), dtype=inputs.dtype)
		b2 = tf.cast(tf.math.greater_equal(inputs, Br), dtype=inputs.dtype)
		b3 = tf.cast(tf.math.logical_and(tf.math.greater_equal(inputs, Bl), tf.math.less(inputs, Br)), dtype=inputs.dtype)

		l1 = (inputs - Bl) * Kl + self.nheight[0]
		l2 = (inputs - Br) * Kr + tf.gather(self.nheight, tf.cast(intervals+1, dtype=tf.int32))
		l3 = (inputs - Bidx_tensor) * Kidx_tensor + Yidx0_tensor

		#Holy shit that was convoluted
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

	def StatisticalAnalysisToggle(self, forceTo=None):
		before = self.collect_stats
		if forceTo != None and type(forceTo) is bool:
			self.collect_stats = forceTo
			return
		else:
			self.collect_stats = not self.collect_stats

		if before and not self.collect_stats: #Meaning we've ended our stats collection phase
			Bl_stat = self.running_avg - 3 * self.running_std
			Br_stat = self.running_avg + 3 * self.running_std
			self.set_weights([self.N_start, np.array([Bl_stat , Br_stat]), np.array([0, 1]), np.linspace(start=Bl_stat, stop=Br_stat, num=self.max_n+1)])
			print("\nRunning Mean: " + str(self.running_avg))
			print("Running Deviation: " + str(self.running_std))

	def ToggleBoundaryLock(self, forceTo=None):
		if forceTo != None and type(forceTo) is bool:
			self.boundary_lock = forceTo
			return
		
		self.boundary_lock = not self.boundary_lock

class NonUniform_PiecewiseLinearUnit(layers.Layer):
	def __init__(self, n=5, momentum=0.9):
		super(PiecewiseLinearUnitV1, self).__init__()
		self.N = n
		
		self.running_avg = 0
		self.running_std = 1
		self.momentum = momentum
		self.collect_stats = False

	def build(self, input_shape):
		self.Bounds = self.add_weight(shape=(2,), initializer='one', trainable=True)
		self.BoundSlope = self.add_weight(shape=(2,), initializer='one', trainable=True)
		self.nheight = self.add_weight(shape=(self.N,), initializer='one', trainable=True)
		self.ratio_points = self.add_weight(shape=(self.N-1,), initializer='one', trainable=True)

		#What I thought was ReLU was not ReLU
		self.set_weights([np.array([-10, 10]), np.array([0, 1]), np.maximum(np.linspace(start=-10, stop=10, num=self.N), 0), np.full(shape=(self.N-1,), fill_value=1)])
	
	def call(self, inputs):
		if self.collect_stats:
			avg = tf.math.reduce_mean(inputs)
			std = tf.math.reduce_std(inputs) #TODO: Rewrite this to avoid "eager execution"

			self.running_avg = (self.running_avg * self.momentum) + (1 - self.momentum) * avg.numpy()
			self.running_std = (self.running_std * self.momentum) + (1 - self.momentum) * std.numpy()

			#Default to ReLU while stats are collected
			b1 = tf.cast(tf.math.greater(inputs, 0.0), dtype=inputs.dtype)

			return b1 * inputs

		def ymxb(m, x, x1, y1):
			return m*x + tf.math.divide_no_nan(y1, m * x1)

		Br, Bl = self.Bounds[1], self.Bounds[0]
		Kr, Kl = self.BoundSlope[1], self.BoundSlope[0]

		interval_base = (Br - Bl)
		interval_ratios = tf.nn.softmax(self.ratio_points)

		bounds = [Bl]
		for i in tf.range(self.N-1):
			bounds.extend([bounds[i] + (interval_base*interval_ratios[i])])

		bound = tf.cast(tf.math.less_equal(inputs, Bl), dtype=inputs.dtype)
		using = bound * inputs
		final_sum = ymxb(Kl, using, Bl, self.nheight[0])

		for i in tf.range(len(bounds)-1):
			fbound = bounds[i]
			tbound = bounds[i+1]
			bound = tf.cast(tf.math.logical_and(tf.math.less_equal(inputs, tbound), tf.math.greater(inputs, fbound)), dtype=inputs.dtype)
			using = bound * inputs
			slope = (self.nheight[i+1] - self.nheight[i]) / (tbound - fbound)
			final_sum += ymxb(slope, using, fbound, self.nheight[i])

		bound = tf.cast(tf.math.greater(inputs, Br), dtype=inputs.dtype)
		using = bound * inputs
		final_sum += ymxb(Kr, using, Br, self.nheight[self.N-1])
	
		return final_sum
	
	def Extract(self):
		print("Boundary Info:\n", 
			"Bl = " + str(self.Bounds[0].numpy()), "w/slope=", str(self.BoundSlope[0].numpy()), "\n",
			"Br = " + str(self.Bounds[1].numpy()), "w/slope=", str(self.BoundSlope[1].numpy()), "\n")
		
		print("All Heights (Interval Count = " + str(self.N) + "): ")
		m = " "
		for i in range(int(self.N + 1)):
			m = m + str(self.nheight[i].numpy()) + ", "
		
		print(m)

	def StatisticalAnalysisToggle(self, forceTo=None):
		before = self.collect_stats
		if forceTo != None and type(forceTo) is bool:
			self.collect_stats = forceTo
			return
		else:
			self.collect_stats = not self.collect_stats

		if before and not self.collect_stats: #Meaning we've ended our stats collection phase
			Bl_stat = self.running_avg - 3 * self.running_std
			Br_stat = self.running_avg + 3 * self.running_std
			self.set_weights([np.array([Bl_stat , Br_stat]), np.array([0, 1]), np.linspace(start=Bl_stat, stop=Br_stat, num=self.N+1)])
			print("\nRunning Mean: " + str(self.running_avg))
			print("Running Deviation: " + str(self.running_std))