import numpy as np
import tensorflow as tf
from .utils import random

#Taken from https://arxiv.org/ftp/arxiv/papers/2108/2108.00700.pdf

def PiLUInit(self, pwlInit):
    self.pwlParams = self.add_weight(shape=(3,), initializer=pwlInit, trainable=True)
    if (self.random_init == False): 
	    self.set_weights([np.array([random(), random(), random()])])
	

def PiLUApply(self, inputs):
	gamma = self.pwlParams[2]
	beta = self.pwlParams[1]
	alpha = self.pwlParams[0]

	b1 = tf.cast(tf.greater(inputs, gamma), inputs.dtype)
	b2 = tf.cast(tf.less_equal(inputs, gamma), inputs.dtype)

	l1 = inputs * alpha + gamma * (1 - alpha)
	l2 = inputs * beta + gamma * (1 - beta)

	return b1*l1 + b2*l2

def PiLUExtract(self):
	gamma = str(self.pwlParams[2].numpy())
	beta = str(self.pwlParams[1].numpy())
	alpha = str(self.pwlParams[0].numpy())
	print("1. " + alpha + "x + " + gamma + "( 1 - " + alpha + ") { " + gamma + " < x }",
          "2. " + beta  + "x + " + gamma + "( 1 - " + beta  + ") { x <= " + gamma + "}",
		  sep="\n")