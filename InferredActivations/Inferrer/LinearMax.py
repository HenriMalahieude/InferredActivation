import tensorflow as tf
from .ActivationLinearizer import ActivationLinearizer
from keras import layers

#This version relates to the Activation Linearizer and having increased amounts of parameters for it
class LinearMaxV1(layers.Layer):
    def __init__(self, divisions=6, max_bounds=6):
        super(LinearMaxV1, self).__init__()
        max_bounds = abs(max_bounds)
        divisions = max(abs(divisions), 2)

        self.exp = ActivationLinearizer(initial_eq="exp", divisions=divisions, left_bound=-1*max_bounds, right_bound=max_bounds)

    def build(self, input_shape):
        self.exp.build(input_shape)

    def call(self, input):
        eexxpp = self.exp(input)
        sum = tf.reduce_sum(eexxpp)
        return eexxpp / sum

#This version just attempts a Rectified Max approach
class LinearMaxV2(layers.Layer):
    def __init__(self):
        super(LinearMaxV2, self).__init__()

    def build(self, input_shape):
        self.exp = tf.nn.relu

    def call(self, input):
        eexxpp = self.exp(input)
        sum = tf.reduce_sum(eexxpp)
        return eexxpp / sum