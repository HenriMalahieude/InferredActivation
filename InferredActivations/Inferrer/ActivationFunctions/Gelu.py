import numpy as np
import tensorflow as tf

#----------GELU
def GeluInit(self, pwlInit):
	self.pwlParams = self.add_weight(shape=(11,), initializer=pwlInit, trainable=True)
	self.set_weights([np.array([-4, -0.795, 0, 0, 0, -0.054, -0.22, 0.22, 0, 1, 0])])
	#I'm just gonna give up doing the random for this one

def GeluApply(self, inputs):
	b1 = tf.cast(tf.math.less_equal(inputs, self.pwlParams[0]), inputs.dtype)
	b2 = tf.cast(tf.math.logical_and(tf.math.greater(inputs, self.pwlParams[0]), tf.math.less_equal(inputs, self.pwlParams[1])))
	b3 = tf.cast(tf.math.logical_and(tf.math.greater(inputs, self.pwlParams[1]), tf.math.less_equal(inputs, self.pwlParams[2])))
	b4 = tf.cast(tf.math.greater(inputs, self.pwlParams[2]))

	l1 = b1 * ((inputs * self.pwlParams[3]) + self.pwlParams[4])
	l2 = tf.math.multiply(b2, tf.math.add(tf.math.multiply(inputs, self.pwlParams[5]), self.pwlParams[6]))
	l3 = tf.math.multiply(b3, tf.math.add(tf.math.multiply(inputs, self.pwlParams[7]), self.pwlParams[8]))
	l4 = tf.math.multiply(b4, tf.math.add(tf.math.multiply(inputs, self.pwlParams[9], self.pwlParams[10])))

	return l1 + l2 + l3 + l4

def GeluExtract(self):
	print("1. y = " + str(self.pwlParams[3]) +"x + " + str(self.pwlParams[4]) +" { x <= " + str(self.pwlParams[0].numpy()) +" }",
       	  "2. y = " + str(self.pwlParams[5].numpy()) + "x + " + str(self.pwlParams[6].numpy()) + " { " + str(self.pwlParams[0].numpy()) + " < x <= " + str(self.pwlParams[1].numpy()) + " }",
		  "3. y = " + str(self.pwlParams[7].numpy()) + "x + " + str(self.pwlParams[8].numpy()) + " { " + str(self.pwlParams[1].numpy()) +" < x <= " + str(self.pwlParams[2].numpy()) +" }",
		  "4. y = " + str(self.pwlParams[9].numpy()) + "x + " + str(self.pwlParams[10].numpy()) + " { x >" + str(self.pwlParams[3]) +" }",
       	  sep="\n")
	return