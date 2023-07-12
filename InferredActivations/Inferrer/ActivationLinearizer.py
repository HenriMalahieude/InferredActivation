import numpy as np
import math
#import tensorflow as tf
from .ActivationFunctions.CustomGradients import LinearBounds
from keras import layers

class ActivationLinearizer(layers.Layer):
    def __init__(self, 
                 initial_eq: str = None, #'random' ; 'sigmoid' ; 'tanh' ; 'gelu'
                 divisions=6, #including outer bounds
                 left_bound=-6,
                 right_bound=6,
                 center_offset=0,
                 interval_length: int = 5 #Encourage a interval length of this
                 ):
        super(ActivationLinearizer, self).__init__()
        assert left_bound < right_bound

        self.pw_count = max(divisions, 2) #why would we ever want to make this less than 2?
        self.center_offset = center_offset
        self.initialization = initial_eq if initial_eq != None else 'sigmoid'
        self.maximum_interval_length = max(abs(interval_length), 0.1)
        self.left_bound = left_bound
        self.right_bound = right_bound

    def build(self, input_shape):
        self.bounds = self.add_weight(shape=(self.pw_count-1,), initializer='ones', trainable=True)
        self.pwlParams = self.add_weight(shape=(self.pw_count*2,), initializer='ones', trainable=True)

        noted_bounds = np.linspace(self.left_bound, self.right_bound, self.pw_count-1) #Bounds could be a param later
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

        print(str(self.pw_count) + ". y = " + str(self.pwlParams[self.pw_count*2-2].numpy()) + "x + " + str(self.pwlParams[self.pw_count*2-1].numpy()) + " { " + str(self.bounds[self.pw_count-2].numpy()) + " < x }")

def MapLinearsBetween(y, x):
    mp = []
    for i in range(len(y)-1):
        m = (y[i+1] - y[i]) / (x[i+1] - x[i])
        b = y[i] - (m * x[i])
        mp.append(m)
        mp.append(b)
    return mp

#I'm sure some function wrapping can make this simpler, but meh
def InitAsSigmoid(self, noted_bounds):
    def Sigmoid(x):
        return 1 / (1 + np.exp(-1 * x))
    
    #self.set_weights([np.array([-6, -3.4, 0, 3.4, 6]), np.array([0, 0, 0.011, 0.071, 0.13705, 0.5, 0.13705, 0.5, 0.011, 0.929, 0, 1])])
    points_to_map = Sigmoid(noted_bounds)
    vals = [0, 0]
    vals.extend(MapLinearsBetween(points_to_map, noted_bounds))
    vals.extend([0, 1])

    #print(vals)
    self.set_weights([noted_bounds, np.array(vals)])

def Tanh(x):
    two_x = np.exp(2*x)
    return (two_x - 1) / (two_x + 1)

def InitAsTanh(self, noted_bounds):
    points_to_map = Tanh(noted_bounds)
    vals = [0, -1]
    vals.extend(MapLinearsBetween(points_to_map, noted_bounds))
    vals.extend([0, 1])

    self.set_weights([noted_bounds, np.array(vals)])

def InitAsExp(self, noted_bounds):
    points_to_map = np.exp(noted_bounds)
    vals = [0, 0]
    vals.extend(MapLinearsBetween(points_to_map, noted_bounds))
    vals.extend(vals[-2:])

    self.set_weights([noted_bounds, np.array(vals)])

def InitAsGelu(self, noted_bounds):
    def GELU(x):
        return (0.5 * x) * ( 1 + Tanh((((2 / math.pi) ** 0.5) * (x + (0.044715 * (x ** 3))))))
    
    points_to_map = GELU(noted_bounds)
    vals = [0, 0]
    vals.extend(MapLinearsBetween(points_to_map, noted_bounds))
    vals.extend([1, 0])

    self.set_weights([noted_bounds, np.array(vals)])

def InitAsRelu(self, noted_bounds):
    vals = [0, 0]
    for i in range(len(noted_bounds)-1):
        if noted_bounds[i] < 0:
            vals.extend([0,0])
        else:
            vals.extend([1,0])
    vals.extend([1, 0])

    self.set_weights([noted_bounds, np.array(vals)])