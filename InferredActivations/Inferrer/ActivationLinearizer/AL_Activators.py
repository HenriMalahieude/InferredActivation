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
def InitAsSigmoid(self, x_arr):
    def Sigmoid(x):
        return 1 / (1 + np.exp(-1 * x))
    
    #self.set_weights([np.array([-6, -3.4, 0, 3.4, 6]), np.array([0, 0, 0.011, 0.071, 0.13705, 0.5, 0.13705, 0.5, 0.011, 0.929, 0, 1])])
    y_arr = Sigmoid(x_arr)
    vals = [0, 0]
    vals.extend(MapLinearsBetween(y_arr, x_arr))
    vals.extend([0, 1])

    #print(vals)
    self.set_weights([x_arr, np.array(vals)])

def Tanh(x):
    two_x = np.exp(2*x)
    return (two_x - 1) / (two_x + 1)

def InitAsTanh(self, x_arr):
    y_arr = Tanh(x_arr)
    vals = [0, -1]
    vals.extend(MapLinearsBetween(y_arr, x_arr))
    vals.extend([0, 1])

    self.set_weights([x_arr, np.array(vals)])

def InitAsExp(self, x_arr):
    y_arr = np.exp(x_arr)
    vals = [0, 0]
    vals.extend(MapLinearsBetween(y_arr, x_arr))
    vals.extend(vals[-2:])

    self.set_weights([x_arr, np.array(vals)])

def InitAsGelu(self, x_arr):
    def GELU(x):
        return (0.5 * x) * ( 1 + Tanh((((2 / math.pi) ** 0.5) * (x + (0.044715 * (x ** 3))))))
    
    y_arr = GELU(x_arr)
    vals = [0, 0]
    vals.extend(MapLinearsBetween(y_arr, x_arr))
    vals.extend([1, 0])

    self.set_weights([x_arr, np.array(vals)])

def InitAsRelu(self, x_arr):
    vals = [0, 0]
    for i in range(len(x_arr)-1):
        if x_arr[i] < 0:
            vals.extend([0,0])
        else:
            vals.extend([1,0])
    vals.extend([1, 0])

    self.set_weights([x_arr, np.array(vals)])

def InitAsReLU6(self, x_arr):
    vals = [0, 0]
    for i in range(len(x_arr)-1):
        if x_arr[i] < 0:
            vals.extend([0, 0])
        elif x_arr[i] >= 6:
            vals.extend([0, 6])
        else:
            vals.extend([1, 0])
    vals.extend([0, 6])

    self.set_weights([x_arr, np.array(vals)])

def InitAsShiftReLU(self, x_arr, shift=-10):
    vals = []
    for i in range(len(x_arr)):
        if x_arr[i] <= shift:
            vals.extend([0, 0])
        else:
            vals.extend([1, shift*-1])

    self.set_weights([x_arr, np.array(vals)])