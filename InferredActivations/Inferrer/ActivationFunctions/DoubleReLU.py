import numpy as np
import tensorflow as tf
from .utils import pseudoInRange

#Taken from https://arxiv.org/ftp/arxiv/papers/2108/2108.00700.pdf

def DoubleReLUInit(self, pwlInit):
    self.pwlParams = self.add_weight(shape=(1,), initializer=pwlInit, trainable=True)
    self.set_weights([np.array([pseudoInRange(0, 4)])])

def DoubleReLUApply(self, inputs): #Unsure if I should prevent the parameter from going negative....
	b1 = tf.cast(tf.math.less(inputs, -1*self.pwlParams[0]), inputs.dtype)
	#b2 = tf.cast(tf.math.logical_and(tf.math.greater(inputs, -1*self.pwlParams[0]), tf.math.less(inputs, self.pwlParams[0])), inputs.dtype)
	b3 = tf.cast(tf.math.greater(inputs, self.pwlParams[0]), inputs.dtype)

	l1 = tf.math.add(inputs, self.pwlParams[0])
	l3 = tf.math.subtract(inputs, self.pwlParams[0])

	return b1*l1 + b3*l3 #Note that we don't need to b2 because it'll be zero anyways

def DoubleReLUExtract(self):
	val = str(self.pwlParams[0].numpy())
	print("1. y = x + " + val + " { x < -" + val +" }",
       	  "2. 0 {-" + val + " <= x <= " + val +"}"
		  "3. y = x - " + val + " { " + val + " < x}", sep="\n")
	return