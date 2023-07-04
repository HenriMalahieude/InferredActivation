import numpy as np
#import tensorflow as tf
from .ActivationFunctions.CustomGradients import LinearBounds
from keras import layers

class ActivationLinearizer(layers.Layer):
    def __init__(self, 
                 initial_eq=None, #'random' ; 'sigmoid'
                 divisions=6, #including outer bounds, overriden if "random" init is not selected
                 center_offset=0):
        super(ActivationLinearizer, self).__init__()
        self.pw_count = max(divisions, 2) #why would we ever want to make this less than 2?
        self.center_offset = center_offset
        self.initialization = initial_eq

    def build(self, input_shape):
        if self.initialization == None or self.initialization == 'sigmoid':
            self.initialization = 'sigmoid' #Redo this to utilize tf.math.sigmoid and stuff to auto generate and not override pw_count
            self.pw_count = 6
        else:
            self.pw_count = self.pw_count #Just noting that there is room for more
        
        self.bounds = self.add_weight(shape=(self.pw_count-1,), initializer='ones', trainable=True)
        self.pwlParams = self.add_weight(shape=(self.pw_count*2,), initializer='ones', trainable=True)

        if self.initialization == 'sigmoid':
            self.set_weights([np.array([-6, -3.4, 0, 3.4, 6]), np.array([0, 0, 0.011, 0.071, 0.13705, 0.5, 0.13705, 0.5, 0.011, 0.929, 0, 1])])
        else:
            if self.initialization != 'random':
                print('ActivationLinearizer: initializer not implemented, defaulting to random')
            
            self.set_weights([np.linspace(-6, 6, self.pw_count-1), np.random.rand(self.pw_count*2)])
    
    def call(self, inputs):
        sum = LinearBounds.OuterBound(self, inputs, paramIndex=[0, 1], boundIndex=0) 
        sum += LinearBounds.OuterBound(self, inputs, paramIndex=[(self.pw_count*2)-2, (self.pw_count*2)-1], boundIndex=(self.pw_count-2), top=True)
        for i in range(self.pw_count-2):
            index_bound = i
            index_slope = index_bound*2 + 2
            sum += LinearBounds.InnerBound(self, inputs, paramIndex=[index_slope, index_slope+1], boundIndex=[index_bound, index_bound+1])
        
        return sum
    
    def extract_linears(self):
        print("1. y = " + str(self.pwlParams[0].numpy()) + "x + " + str(self.pwlParams[1].numpy()) + " { x <= " + str(self.bounds[0].numpy()) + " }")

        for i in range(self.pw_count - 2):
            index_bound = i
            index_slope = i*2 + 2
            print(str(i+2) + ". y = " + str(self.pwlParams[index_slope].numpy()) + "x + " + str(self.pwlParams[index_slope+1].numpy()) + " { " + str(self.bounds[index_bound].numpy()) + " < x <= " + str(self.bounds[index_bound+1].numpy()) + " }")

        print(str(self.pw_count) + ". y = " + str(self.pwlParams[self.pw_count*2-2].numpy()) + "x + " + str(self.pwlParams[self.pw_count*2-1].numpy()) + " { x <= " + str(self.bounds[self.pw_count-2].numpy()) + " }")
