import tensorflow as tf
from .ActivationLinearizer import ActivationLinearizer
from .PolynomialActivation import SLAF_PolynomialActivation
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
        eexxpp = tf.nn.relu(self.exp(input)) #Without Relu: is V1, with: is 1.5
        sum = tf.reduce_sum(eexxpp)
        return eexxpp / sum
    
    def Extract(self):
        self.exp.extract_linears()

#This version just attempts a Rectified Max approach
#NOTE: It's bad
class LinearMaxV2(layers.Layer):
    def __init__(self):
        super(LinearMaxV2, self).__init__()

    def build(self, input_shape):
        self.exp = tf.nn.relu

    def call(self, input):
        eexxpp = self.exp(input)
        sum = tf.reduce_sum(eexxpp)
        return eexxpp / sum

#This version attempts a simplified exponentiator
#Inspired by https://www.nature.com/articles/s41598-021-94691-7
#NOTE: It's okay
class LinearMaxV3(layers.Layer):
    def __init__(self):
        super(LinearMaxV3, self).__init__()

    def build(self, input_shape):
        self.base = self.add_weight(shape=(), initializer='one', trainable=True)

    def call(self, input):
        eexxpp = tf.math.pow(self.base, input)
        sum = tf.reduce_sum(eexxpp)
        return eexxpp / sum

#Messing around
#NOTE: It's.... okay? only v1 tho
class LinearMaxV4(layers.Layer):
    def __init__(self):
        super(LinearMaxV4, self).__init__()

    def build(self, input_shape):
        self.exp = SLAF_PolynomialActivation(degree=3, bounds=False)
        self.exp.build(input_shape)

    def call(self, input):
        eexxpp = tf.nn.relu(self.exp(input)) #w/o relu V1, w/ V1.5
        sum = tf.reduce_sum(eexxpp)
        return eexxpp / sum
