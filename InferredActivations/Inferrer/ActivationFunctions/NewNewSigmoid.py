#from .utils import pseudoInRange, random
import tensorflow as tf
import numpy as np

def NNSigInit(self, pwlInit):
    self.bounds = self.add_weight(shape=(5,), initializer=pwlInit, trainable=True)
    self.pwlParams = self.add_weight(shape=(12,), initializer=pwlInit, trainable=True)
    self.set_weights([np.array([-6, -3.4, 0, 3.4, 6]), np.array([0, 0.011, 0.071, 0.75, 2, 0.5, 0.75, 2, 0.5, 0.011, 0.929, 1])])

    if (self.random_init):
        print("NewNewSigmoid does not support randomization") #Someone else can do this, I've got better things to do (future me be like "...")
    return

def NNSigExtract(self):
    print("1. y = " + str(self.pwlParams[0].numpy()) + "x { x <= " + str(self.bounds[0].numpy()) +" }",
       	  "2. y = " + str(self.pwlParams[1].numpy()) + "x + " + str(self.pwlParams[2].numpy()) + " { " + str(self.bounds[0].numpy()) + " < x <= " + str(self.bounds[1].numpy()) + " }",
		  "3. y = ((" + str(self.pwlParams[3].numpy()) + "x) / (" + str(self.pwlParams[4].numpy()) + " - x)) + " + str(self.pwlParams[5].numpy()) + " { " + str(self.bounds[1].numpy()) +" < x <= " + str(self.bounds[2].numpy()) +" }",
		  "4. y = ((" + str(self.pwlParams[6].numpy()) + "x) / (" + str(self.pwlParams[7].numpy()) + " + x)) + " + str(self.pwlParams[8].numpy()) + " { " + str(self.bounds[2].numpy()) +" < x <= " + str(self.bounds[3].numpy()) +" }",
       	  "5. y = " + str(self.pwlParams[9].numpy()) + "x + " + str(self.pwlParams[10].numpy()) + " { " + str(self.bounds[3].numpy()) + " < x <= " + str(self.bounds[4].numpy()) + " }",
		  "6. y = " + str(self.pwlParams[11].numpy()) + "x { x > " + str(self.bounds[4].numpy()) +" }", sep="\n")

#This one will be big, so move to bottom
def NNSigApply(self, inputs):
    #b1 = tf.cast(tf.math.less_equal(inputs, self.bounds[0]), inputs.dtype)
    #l1 = tf.math.multiply(b1, self.pwlParams[0])
    @tf.custom_gradient
    def E1(x): #Bottom Bound
        bound = tf.cast(tf.math.less_equal(x, self.bounds[0]), x.dtype)
        y = bound * x * self.pwlParams[0]
        def gradient_function(dx, variables): #Note Variables argument is keyword required
            dy_dx = bound * self.pwlParams[0]
            dscalar = tf.reshape(tf.reduce_mean(bound * dx), [1])
            grad_vars = [tf.pad(dscalar, [[0, 4]]), tf.pad(dscalar, [[0, variables[1].shape.as_list()[0]-1]])]
            #print("E1:", dy_dx, grad_vars[0], grad_vars[1], sep="\n\t")
            return dy_dx, grad_vars
        
        return y, gradient_function

    #b2 = tf.cast(tf.math.logical_and(tf.math.greater(inputs, self.bounds[0]), tf.math.less_equal(inputs, self.bounds[1])), inputs.dtype)
    #l2 = tf.math.multiply(b2, tf.math.add(tf.math.multiply(inputs, self.pwlParams[1]), self.pwlParams[2]))
    @tf.custom_gradient
    def E2(x):
        bound = tf.cast(tf.math.logical_and(tf.math.greater(x, self.bounds[0]), tf.math.less_equal(x, self.bounds[1])), x.dtype)
        y = tf.math.multiply(bound, tf.math.add(tf.math.multiply(x, self.pwlParams[1]), self.pwlParams[2]))
        def grad_fn(dx, variables):
            dy_dx = bound * self.pwlParams[1] #Debatable whether I should put pwlParams[2] in here
            dscalar = tf.constant([1, 1], dtype=x.dtype) * tf.reshape(tf.reduce_mean(bound * dx), [1]) #Duplicating the value
            dbounds = tf.pad(dscalar, [[0, 3]])
            dparams = tf.pad(dscalar, [[1, variables[1].shape.as_list()[0]-3]])
            #print("E2:", dy_dx, dbounds, dparams, sep="\n\t")
            return dy_dx, [dbounds, dparams]
        return y, grad_fn

	#b3 = tf.cast(tf.math.logical_and(tf.math.greater(inputs, self.bounds[1]), tf.math.less_equal(inputs, self.bounds[2])), inputs.dtype)
	#l3 = tf.math.multiply(b3, tf.math.add(tf.math.divide_no_nan(tf.math.multiply(inputs, self.pwlParams[3]), tf.math.subtract(inputs, self.pwlParams[4])), self.pwlParams[5]))
    @tf.custom_gradient
    def E3(x):
        bound = tf.cast(tf.math.logical_and(tf.math.greater(x, self.bounds[1]), tf.math.less_equal(x, self.bounds[2])), x.dtype)
        y = tf.math.multiply(bound, tf.math.add(tf.math.divide_no_nan(tf.math.multiply(x, self.pwlParams[3]), tf.math.subtract(x, self.pwlParams[4])), self.pwlParams[5]))
        def grad_fn(dx, variables):
            dy_dx = bound * tf.math.divide_no_nan((self.pwlParams[3]*2), tf.pow((dx - self.pwlParams[4]), 2)) #manually calculated derivative
            dscalar = tf.reshape(tf.reduce_mean(bound * dx), [1])
            dscalar1 = tf.constant([1, 1], dtype=x.dtype) * dscalar #Duplicating the value
            dscalar2 = tf.constant([1, 1, 1], dtype=x.dtype) * dscalar
            dbounds = tf.pad(dscalar1, [[1, 2]])
            dparams = tf.pad(dscalar2, [[3, variables[1].shape.as_list()[0]-6]])
            #print("E3:", dy_dx, dbounds, dparams, sep="\n\t")
            return dy_dx, [dbounds, dparams]
        return y, grad_fn

    #b4 = tf.cast(tf.math.logical_and(tf.math.greater(inputs, self.bounds[2]), tf.math.less_equal(inputs, self.bounds[3])), inputs.dtype)
    #l4 = tf.math.multiply(b4, tf.math.add(tf.math.divide_no_nan(tf.math.multiply(inputs, self.pwlParams[6]), tf.math.add(inputs, self.pwlParams[7])), self.pwlParams[8]))
    @tf.custom_gradient
    def E4(x):
        bound = tf.cast(tf.math.logical_and(tf.math.greater(x, self.bounds[2]), tf.math.less_equal(x, self.bounds[3])), x.dtype)
        y = tf.math.multiply(bound, tf.math.add(tf.math.divide_no_nan(tf.math.multiply(x, self.pwlParams[6]), tf.math.add(x, self.pwlParams[7])), self.pwlParams[8]))
        def grad_fn(dx, variables):
            dy_dx = bound * tf.math.divide_no_nan((self.pwlParams[6]*2), tf.pow((dx - self.pwlParams[7]), 2)) #manually calculated derivative
            dscalar = tf.reshape(tf.reduce_mean(bound * dx), [1])
            dscalar1 = tf.constant([1, 1], dtype=x.dtype) * dscalar #Duplicating the value
            dscalar2 = tf.constant([1, 1, 1], dtype=x.dtype) * dscalar
            dbounds = tf.pad(dscalar1, [[2, 1]])
            dparams = tf.pad(dscalar2, [[6, variables[1].shape.as_list()[0]-9]])
            #print("E4:", dy_dx, dbounds, dparams, sep="\n\t")
            return dy_dx, [dbounds, dparams]
        return y, grad_fn
    
    #b5 = tf.cast(tf.math.logical_and(tf.math.greater(inputs, self.bounds[3]), tf.math.less_equal(inputs, self.bounds[4])), inputs.dtype)
    #l5 = tf.math.multiply(b5, tf.math.add(tf.math.multiply(inputs, self.pwlParams[9]), self.pwlParams[10]))
    @tf.custom_gradient
    def E5(x):
        bound = tf.cast(tf.math.logical_and(tf.math.greater(x, self.bounds[3]), tf.math.less_equal(x, self.bounds[4])), x.dtype)
        y = tf.math.multiply(bound, tf.math.add(tf.math.multiply(x, self.pwlParams[9]), self.pwlParams[10]))
        def grad_fn(dx, variables):
            dy_dx = bound * self.pwlParams[9]
            dscalar = tf.constant([1, 1], dtype=x.dtype) * tf.reshape(tf.reduce_mean(bound * dx), [1]) #Duplicating the value
            dbounds = tf.pad(dscalar, [[3, 0]])
            dparams = tf.pad(dscalar, [[9, variables[1].shape.as_list()[0]-11]])
            #print("E5:", dy_dx, dbounds, dparams, sep="\n\t")
            return dy_dx, [dbounds, dparams]
        return y, grad_fn

    #b6 = tf.cast(tf.math.greater(inputs, self.bounds[4]), inputs.dtype)
	#l6 = tf.math.multiply(b6, self.pwlParams[11])
    @tf.custom_gradient
    def E6(x): #Top Bound
        bound = tf.cast(tf.math.less_equal(x, self.bounds[4]), x.dtype)
        y = bound * x * self.pwlParams[11]
        def grad_fn(dx, variables):
            dy_dx = bound * self.pwlParams[11]
            dscalar = tf.reshape(tf.reduce_mean(bound * dx), [1])
            grad_vars = [tf.pad(dscalar, [[4, 0]]), tf.pad(dscalar, [[variables[1].shape.as_list()[0]-1, 0]])]
            #print("E6:", dy_dx, grad_vars[0], grad_vars[1], sep="\n\t")
            return dy_dx, grad_vars
        
        return y, grad_fn

    return E1(inputs) + E2(inputs) + E3(inputs) + E4(inputs) + E5(inputs) + E6(inputs)