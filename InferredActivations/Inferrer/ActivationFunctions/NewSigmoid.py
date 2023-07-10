import numpy as np
import tensorflow as tf
from .utils import pseudoInRange, random

#NOTE: It seems that there needs to be more than just the bounds for calculating gradients

@tf.custom_gradient
def CatchGradientFor(bound):
	def Grad(upstream):
		return upstream * bound
	return bound, Grad

#Testing Custom Gradients
def LowerWrapper(x, bound):
	@tf.custom_gradient
	def Internal(b):
		thing = tf.cast(tf.math.less_equal(x, b), x.dtype)
		def grad(upstream):
			return upstream * b
		return thing, grad
	return Internal(bound)

def InnerWrapper(x, blow, bhigh):
	@tf.custom_gradient
	def Internal(b1, b2):
		thing = tf.cast(tf.math.logical_and(tf.math.greater(x, b1), tf.math.less_equal(x, b2)), dtype=x.dtype)
		id = tf.cast(tf.math.greater(thing, -5), x.dtype) #See, this is some ingenuity right here
		def grad(upstream):
			return upstream * b1 * id, upstream * b2 * id
		return thing, grad
	return Internal(blow, bhigh)

def UpperWrapper(x, bound):
	@tf.custom_gradient
	def Internal(b):
		thing = tf.cast(tf.math.greater(x, b), x.dtype)
		def grad(upstream):
			return upstream * b * tf.cast(tf.math.greater(thing, -5), x.dtype) #See, this is some ingenuity right here
		return thing, grad
	return Internal(bound)

#Fuck it we ball

#----------Sigmoid
def NSInit(self, pwlInit):
	self.bounds = self.add_weight(shape=(5,), initializer=pwlInit, trainable=True)
	self.pwlParams = self.add_weight(shape=(12,), initializer=pwlInit, trainable=True)
	if (self.random_init != True):
		self.set_weights([np.array([-6, -3.4, 0, 3.4, 6]), np.array([0, 0.011, 0.071, 0.75, 2, 0.5, 0.75, 2, 0.5, 0.011, 0.929, 1])])
	else:
		b1 = pseudoInRange(-7, -5)
		b2 = pseudoInRange(b1, -3)
		b3 = pseudoInRange(b2, 0)
		b4 = pseudoInRange(b3, 3)
		b5 = pseudoInRange(b4, 7)
		print("Randomly Set Boundaries to:", b1, b2, b3, b4, b5, sep=" ")
		self.set_weights([np.array([b1, b2, b3, b4, b5]), np.array([random() - 1,random(),random(),random(),random(),random(),random(),random(),random(),random(),random(),random()])])

def NSApply(self, inputs):
	b1 = LowerWrapper(inputs, self.bounds[0])
	b2 = InnerWrapper(inputs, self.bounds[0], self.bounds[1])
	b3 = InnerWrapper(inputs, self.bounds[1], self.bounds[2]) #tf.cast(tf.math.logical_and(tf.math.greater(inputs, self.pwlParams[1]), tf.math.less_equal(inputs, self.pwlParams[2])), inputs.dtype)
	b4 = InnerWrapper(inputs, self.bounds[2], self.bounds[3]) #tf.cast(tf.math.logical_and(tf.math.greater(inputs, self.pwlParams[2]), tf.math.less_equal(inputs, self.pwlParams[3])), inputs.dtype)
	b5 = InnerWrapper(inputs, self.bounds[3], self.bounds[4]) #tf.cast(tf.math.logical_and(tf.math.greater(inputs, self.pwlParams[3]), tf.math.less_equal(inputs, self.pwlParams[4])), inputs.dtype)
	b6 = UpperWrapper(inputs, self.bounds[4]) #tf.cast(tf.math.greater(inputs, self.pwlParams[4]), inputs.dtype)

	#Calculate Each Linear Piece
	l1 = b1 * self.pwlParams[0]
	l2 = b2 * ((inputs * self.pwlParams[1]) + self.pwlParams[2])
	l3 = b3 * (tf.math.divide_no_nan((inputs * self.pwlParams[3]), tf.math.subtract(inputs, self.pwlParams[4])) + self.pwlParams[5])
	l4 = b4 * (tf.math.divide_no_nan((inputs * self.pwlParams[6]), tf.math.subtract(inputs, self.pwlParams[7])) + self.pwlParams[8])
	l5 = b5 * ((inputs * self.pwlParams[9]) + self.pwlParams[10])
	l6 = b6 * self.pwlParams[11]

	return l1 + l2 + l3 + l4 + l5 + l6

def NSExtract(self):
	print("1. y = " + str(self.pwlParams[0].numpy()) + "x { x <= " + str(self.bounds[0].numpy()) +" }",
       	  "2. y = " + str(self.pwlParams[1].numpy()) + "x + " + str(self.pwlParams[2].numpy()) + " { " + str(self.bounds[0].numpy()) + " < x <= " + str(self.bounds[1].numpy()) + " }",
		  "3. y = ((" + str(self.pwlParams[3].numpy()) + "x) / (" + str(self.pwlParams[4].numpy()) + " - x)) + " + str(self.pwlParams[5].numpy()) + " { " + str(self.bounds[1].numpy()) +" < x <= " + str(self.bounds[2].numpy()) +" }",
		  "4. y = ((" + str(self.pwlParams[6].numpy()) + "x) / (" + str(self.pwlParams[7].numpy()) + " + x)) + " + str(self.pwlParams[8].numpy()) + " { " + str(self.bounds[2].numpy()) +" < x <= " + str(self.bounds[3].numpy()) +" }",
       	  "5. y = " + str(self.pwlParams[9].numpy()) + "x + " + str(self.pwlParams[10].numpy()) + " { " + str(self.bounds[3].numpy()) + " < x <= " + str(self.bounds[4].numpy()) + " }",
		  "6. y = " + str(self.pwlParams[11].numpy()) + "x { x > " + str(self.bounds[4].numpy()) +" }", sep="\n")
	return