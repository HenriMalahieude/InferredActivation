import numpy as np
import tensorflow as tf
from .LinearBounds import OuterBoundUnlocked, InnerBoundUnlocked
from .AL_Activators import *
from keras import layers

#NOTE: pwlParams is organized where [slope, bias, slope2, bias2, ..., etc]
#NOTE: bounds is organized where [b1, b2, b3, ..., etc] and are simple floats marking boundaries

@tf.keras.saving.register_keras_serializable('InferredActivation')
class ActivationLinearizer(layers.Layer):
    def __init__(self, 
                 initial_eq: str = None, #'random' ; 'sigmoid' ; 'tanh' ; 'gelu' ; 'exp' ; 'relu'
                 lock_boundaries: bool = False,
                 divisions: float = 6, #including outer bounds
                 left_bound: float = -6,
                 right_bound: float = 6,
                 center_offset: float = 0,
                 interval_length: int = 5 #Encourage a interval length of this
                 ):
        super(ActivationLinearizer, self).__init__()
        assert left_bound < right_bound

        self.pw_count = max(divisions, 2) #why would we ever want to make this less than 2?
        self.center_offset = center_offset
        self.initialization = initial_eq if initial_eq != None else 'relu'
        self.maximum_interval_length = max(abs(interval_length), 0.1)
        self.left_bound = left_bound
        self.right_bound = right_bound
        self.boundary_lock = lock_boundaries

    def get_config(self):
        base_config = super().get_config()
        config = {
            "pw_count": self.pw_count, 
            "center_offset": self.center_offset, 
            "initialization": self.initialization, 
            "maximum_interval_length": self.maximum_interval_length,
            "left_bound": self.left_bound,
            "right_bound": self.right_bound
        }

        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return ActivationLinearizer(config["initialization"], config["pw_count"], config["left_bound"], config["right_bound"], config["center_offset"], config["maximum_interval_length"])

    def build(self, input_shape):
        self.bounds = self.add_weight(shape=(self.pw_count-1,), initializer='ones', trainable=True)
        self.pwlParams = self.add_weight(shape=(self.pw_count*2,), initializer='ones', trainable=True)

        noted_bounds = np.linspace(self.left_bound, self.right_bound, self.pw_count-1) #Bounds could be an init param later
        self.set_weights([noted_bounds, np.random.randn(self.pw_count*2)])

        if self.initialization == 'sigmoid':
            InitAsSigmoid(self, noted_bounds)
        elif self.initialization == 'tanh':
            InitAsTanh(self, noted_bounds)
        elif self.initialization == 'exp':
            InitAsExp(self, noted_bounds)
        elif self.initialization == 'gelu':
            InitAsGelu(self, noted_bounds)
        elif self.initialization == 'relu':
            InitAsRelu(self, noted_bounds)
        elif self.initialization != 'random':
            print('ActivationLinearizer: initializer not implemented, defaulting to random')    
    
    def _call_unlock(self, inputs):
        sum = OuterBoundUnlocked(self, inputs, paramIndex=[0, 1], boundIndex=0) 
        sum += OuterBoundUnlocked(self, inputs, paramIndex=[(self.pw_count*2)-2, (self.pw_count*2)-1], boundIndex=(self.pw_count-2), top=True)
        for i in range(self.pw_count-2):
            index_bound = i
            index_slope = index_bound*2 + 2
            sum += InnerBoundUnlocked(self, inputs, paramIndex=[index_slope, index_slope+1], boundIndex=[index_bound, index_bound+1])
        
        return sum
    
    def _call_lock(self, inputs):
        bounds_calced = [OuterLockedBound(inputs, self.bounds[0])]
        for i in range(self.pw_count-2):
            index = i
            bounds_calced.append(InnerLockedBound(inputs, self.bounds[index], self.bounds[index+1]))

        bounds_calced.append(OuterLockedBound(inputs, self.bounds[self.pw_count-1]))

        output = (bounds_calced[0] * inputs) * self.pwlParams[0] + self.pwlParams[1]
        output += (bounds_calced[-1] * inputs) * self.pwlParams[-2] + self.pwlParams[-1]
        for i in range(self.pw_count-2):
            index_bound = i
            index_slope = index_bound*2 + 2
            output += (bounds_calced[index_bound] * inputs) * self.pwlParams[index_slope] + self.pwlParams[index_slope+1]

        return output
    
    def call(self, inputs):
        if self.boundary_lock:
            return self._call_lock(self, inputs)
        else:
            return self._call_unlock(self, inputs)
    
    def lock_boundaries(self, force_to = None):
        if force_to != None:
            self.boundary_lock = force_to
        self.boundary_lock = not self.boundary_lock

    def extract_linears(self):
        print("1. y = " + str(self.pwlParams[0].numpy()) + "x + " + str(self.pwlParams[1].numpy()) + " { x <= " + str(self.bounds[0].numpy()) + " }")

        for i in range(self.pw_count - 2):
            index_bound = i
            index_slope = i*2 + 2
            print(str(i+2) + ". y = " + str(self.pwlParams[index_slope].numpy()) + "x + " + str(self.pwlParams[index_slope+1].numpy()) + " { " + str(self.bounds[index_bound].numpy()) + " < x <= " + str(self.bounds[index_bound+1].numpy()) + " }")

        print(str(self.pw_count) + ". y = " + str(self.pwlParams[self.pw_count*2-2].numpy()) + "x + " + str(self.pwlParams[self.pw_count*2-1].numpy()) + " { " + str(self.bounds[self.pw_count-2].numpy()) + " < x }")

def OuterLockedBound(inputs, bound, top=True):
    func = tf.math.less_equal if top else tf.math.greater
    return tf.cast(func(inputs, bound), inputs.dtype)

def InnerLockedBound(inputs, bound1, bound2):
    assert bound1 < bound2
    return tf.cast(tf.math.logical_and(tf.math.greater(inputs, bound1), tf.math.less_equal(inputs, bound2)), inputs.dtype)