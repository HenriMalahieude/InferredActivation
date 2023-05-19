import numpy as np
import tensorflow as tf
from .utils import pseudoInRange, random

#----------Tanh
def TanhInit(self, pwlInit):
	self.pwlParams = self.add_weight(shape=(13,), initializer=pwlInit, trainable=True)
	if (self.random_init == False):
		self.set_weights([np.array([-3, -1.5, 0, 1.5, 3, 0.05, -0.85, 0.61, 0, 0.61, 0, 0.05, 0.85])])
	else:
		b1 = pseudoInRange(-4, -2)
		b2 = pseudoInRange(b1, -1)
		b3 = pseudoInRange(b2, 0)
		b4 = pseudoInRange(b3, 1)
		b5 = pseudoInRange(b4, 4)
		print("Randomly Set Boundaries to:", b1, b2, b3, b4, b5, sep=" ")
		self.set_weights([np.array([b1, b2, b3, b4, b5, random(),random()-1,random(),random(),random(),random(),random(),random()])])
	return

def TanhApply(self, inputs):
	b1 = tf.cast(tf.math.less_equal(inputs, self.pwlParams[0]), inputs.dtype)
	b2 = tf.cast(tf.math.logical_and(tf.math.greater(inputs, self.pwlParams[0]), tf.math.less_equal(inputs, self.pwlParams[1])), inputs.dtype)
	b3 = tf.cast(tf.math.logical_and(tf.math.greater(inputs, self.pwlParams[1]), tf.math.less_equal(inputs, self.pwlParams[2])), inputs.dtype)
	b4 = tf.cast(tf.math.logical_and(tf.math.greater(inputs, self.pwlParams[2]), tf.math.less_equal(inputs, self.pwlParams[3])), inputs.dtype)
	b5 = tf.cast(tf.math.logical_and(tf.math.greater(inputs, self.pwlParams[3]), tf.math.less_equal(inputs, self.pwlParams[4])), inputs.dtype)
	b6 = tf.cast(tf.math.greater(inputs, self.pwlParams[4]), inputs.dtype)

	l1 = tf.math.multiply(b1, -1)
	l2 = tf.math.multiply(b2, tf.math.add(tf.math.multiply(inputs, self.pwlParams[5]), self.pwlParams[6]))
	l3 = tf.math.multiply(b3, tf.math.add(tf.math.multiply(inputs, self.pwlParams[7]), self.pwlParams[8]))
	l4 = tf.math.multiply(b4, tf.math.add(tf.math.multiply(inputs, self.pwlParams[9]), self.pwlParams[10]))
	l5 = tf.math.multiply(b5, tf.math.add(tf.math.multiply(inputs, self.pwlParams[11]), self.pwlParams[12]))
	l6 = tf.math.multiply(b6, 1)
	
	return l1 + l2 + l3 + l4 + l5 + l6

def TanhExtract(self):
	print("1. y = -1 { x <= " + str(self.pwlParams[0].numpy()) +" }",
       	  "2. y = " + str(self.pwlParams[5].numpy()) + "x + " + str(self.pwlParams[6].numpy()) + " { " + str(self.pwlParams[0].numpy()) + " < x <= " + str(self.pwlParams[1].numpy()) + " }",
		  "3. y = " + str(self.pwlParams[7].numpy()) + "x + " + str(self.pwlParams[8].numpy()) + " { " + str(self.pwlParams[1].numpy()) +" < x <= " + str(self.pwlParams[2].numpy()) +" }",
		  "4. y = " + str(self.pwlParams[9].numpy()) + "x + " + str(self.pwlParams[10].numpy()) + " { " + str(self.pwlParams[2].numpy()) +" < x <= " + str(self.pwlParams[3].numpy()) +" }",
       	  "5. y = " + str(self.pwlParams[11].numpy()) + "x + " + str(self.pwlParams[12].numpy()) + " { " + str(self.pwlParams[3].numpy()) + " < x <= " + str(self.pwlParams[4].numpy()) + " }",
		  "6. y = 1 { x > " + str(self.pwlParams[4].numpy()) +" }", sep="\n")
	return