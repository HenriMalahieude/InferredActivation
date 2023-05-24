from keras import layers
from .ActivationFunctions import SigApproximator

#Now that Approximators are wrapped in a meta EqAproximator function, there should no longer need to be any edits to this
class InferredActivation(layers.Layer):
	def __init__(self, randomize=False, eq_funcs=SigApproximator):
		super(InferredActivation, self).__init__()
		self.func = eq_funcs
		self.random_init = randomize

	def get_config(self):
		config = super().get_config()
		config.update({
			"randomize": self.random_init,
			"eq_funcs": self.func,
			"random_init": self.random_init,
			"func": self.func
		})
	
	def build(self, input_shape):
		pwlInit = ('random_normal' if self.random_init else 'one')
		self.func(self, 'init', pwlInit)

	def call(self, inputs):
		return self.func(self, 'apply', inputs)

	def Extract(self):
		self.func(self, 'extract')