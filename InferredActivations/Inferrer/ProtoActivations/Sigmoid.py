import numpy as np
import tensorflow as tf
from .utils import pseudoInRange, random

#----------Sigmoid
#NOTE: Currently the Sigmoid Approximater uses the PWL that has (2-x) in the denom (hence the divide_no_nan), I feel like this could be more effective with something else
#I'm sure someone with more python experience could make this better
def SigInit(self, pwlInit):
	self.pwlParams = self.add_weight(shape=(17,), initializer=pwlInit, trainable=True)
	if (self.random_init != True):
		self.set_weights([np.array([-6, -3.4, 0, 3.4, 6, 0, 0.011, 0.071, 0.75, 2, 0.5, 0.75, 2, 0.5, 0.011, 0.929, 1])])
	else:
		b1 = pseudoInRange(-7, -5)
		b2 = pseudoInRange(b1, -3)
		b3 = pseudoInRange(b2, 0)
		b4 = pseudoInRange(b3, 3)
		b5 = pseudoInRange(b4, 7)
		print("Randomly Set Boundaries to:", b1, b2, b3, b4, b5, sep=" ")
		self.set_weights([np.array([b1, b2, b3, b4, b5, random() - 1,random(),random(),random(),random(),random(),random(),random(),random(),random(),random(),random()])])

def SigApply(self, inputs):
	#Start the Piece-Wise Linear thingy (let us commence forth)
	#Calculate Boundaries... With this much math, it makes me wonder if this will really increase efficiency (compared to a single calc for all of the tensor by sigmoid)
	b1 = tf.cast(tf.math.less_equal(inputs, self.pwlParams[0]), inputs.dtype)
	b2 = tf.cast(tf.math.logical_and(tf.math.greater(inputs, self.pwlParams[0]), tf.math.less_equal(inputs, self.pwlParams[1])), inputs.dtype)
	b3 = tf.cast(tf.math.logical_and(tf.math.greater(inputs, self.pwlParams[1]), tf.math.less_equal(inputs, self.pwlParams[2])), inputs.dtype)
	b4 = tf.cast(tf.math.logical_and(tf.math.greater(inputs, self.pwlParams[2]), tf.math.less_equal(inputs, self.pwlParams[3])), inputs.dtype)
	b5 = tf.cast(tf.math.logical_and(tf.math.greater(inputs, self.pwlParams[3]), tf.math.less_equal(inputs, self.pwlParams[4])), inputs.dtype)
	b6 = tf.cast(tf.math.greater(inputs, self.pwlParams[4]), inputs.dtype)

	#Calculate Each Linear Piece
	l1 = tf.math.multiply(b1, self.pwlParams[5])
	l2 = tf.math.multiply(b2, tf.math.add(tf.math.multiply(inputs, self.pwlParams[6]), self.pwlParams[7]))
	l3 = tf.math.multiply(b3, tf.math.add(tf.math.divide_no_nan(tf.math.multiply(inputs, self.pwlParams[8]), tf.math.subtract(inputs, self.pwlParams[9])), self.pwlParams[10]))
	l4 = tf.math.multiply(b4, tf.math.add(tf.math.divide_no_nan(tf.math.multiply(inputs, self.pwlParams[11]), tf.math.add(inputs, self.pwlParams[12])), self.pwlParams[13]))
	l5 = tf.math.multiply(b5, tf.math.add(tf.math.multiply(inputs, self.pwlParams[14]), self.pwlParams[15]))
	l6 = tf.math.multiply(b6, self.pwlParams[16])

	return l1 + l2 + l3 + l4 + l5 + l6

def SigExtract(self):
	print("1. y = " + str(self.pwlParams[5].numpy()) + "x { x <= " + str(self.pwlParams[0].numpy()) +" }",
       	  "2. y = " + str(self.pwlParams[6].numpy()) + "x + " + str(self.pwlParams[7].numpy()) + " { " + str(self.pwlParams[0].numpy()) + " < x <= " + str(self.pwlParams[1].numpy()) + " }",
		  "3. y = ((" + str(self.pwlParams[8].numpy()) + "x) / (" + str(self.pwlParams[9].numpy()) + " - x)) + " + str(self.pwlParams[10].numpy()) + " { " + str(self.pwlParams[1].numpy()) +" < x <= " + str(self.pwlParams[2].numpy()) +" }",
		  "4. y = ((" + str(self.pwlParams[11].numpy()) + "x) / (" + str(self.pwlParams[12].numpy()) + " + x)) + " + str(self.pwlParams[13].numpy()) + " { " + str(self.pwlParams[2].numpy()) +" < x <= " + str(self.pwlParams[3].numpy()) +" }",
       	  "5. y = " + str(self.pwlParams[14].numpy()) + "x + " + str(self.pwlParams[15].numpy()) + " { " + str(self.pwlParams[3].numpy()) + " < x <= " + str(self.pwlParams[4].numpy()) + " }",
		  "6. y = " + str(self.pwlParams[16].numpy()) + "x { x > " + str(self.pwlParams[4].numpy()) +" }", sep="\n")
	return


#NOTE: For testing Boundary learning only
def SigBoundaryApply(self, inputs):
	b1 = tf.cast(tf.math.less_equal(inputs, self.pwlParams[0]), inputs.dtype)
	b2 = tf.cast(tf.math.logical_and(tf.math.greater(inputs, self.pwlParams[0]), tf.math.less_equal(inputs, self.pwlParams[1])), inputs.dtype)
	b3 = tf.cast(tf.math.logical_and(tf.math.greater(inputs, self.pwlParams[1]), tf.math.less_equal(inputs, self.pwlParams[2])), inputs.dtype)
	b4 = tf.cast(tf.math.logical_and(tf.math.greater(inputs, self.pwlParams[2]), tf.math.less_equal(inputs, self.pwlParams[3])), inputs.dtype)
	b5 = tf.cast(tf.math.logical_and(tf.math.greater(inputs, self.pwlParams[3]), tf.math.less_equal(inputs, self.pwlParams[4])), inputs.dtype)
	b6 = tf.cast(tf.math.greater(inputs, self.pwlParams[4]), inputs.dtype)

	l1 = tf.math.multiply(b1, 0)
	l2 = tf.math.multiply(b2, tf.math.add(tf.math.multiply(inputs, 0.011), 0.071))
	l3 = tf.math.multiply(b3, tf.math.add(tf.math.divide_no_nan(tf.math.multiply(inputs, 0.75), tf.math.subtract(inputs, 2)), 0.5))
	l4 = tf.math.multiply(b4, tf.math.add(tf.math.divide_no_nan(tf.math.multiply(inputs, 0.75), tf.math.add(inputs, 2)), 0.5))
	l5 = tf.math.multiply(b5, tf.math.add(tf.math.multiply(inputs, 0.011), 0.929))
	l6 = tf.math.multiply(b6, 1)

	return l1 + l2 + l3 + l4 + l5 + l6