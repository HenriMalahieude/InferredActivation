import math
import numpy as np

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