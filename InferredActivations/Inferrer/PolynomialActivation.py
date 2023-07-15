import tensorflow as tf
from keras import layers

#From https://arxiv.org/pdf/1906.09529.pdf
class SLAF_PolynomialActivation(layers.Layer):
    def __init__(self, degree=3, max_val=100, min_val=-100):
        super(SLAF_PolynomialActivation, self).__init__()
        assert min_val < max_val
        
        self.degree = max(degree, 2)
        self.max_val = min(max_val, 1000)
        self.min_val = max(min_val, -1000)

    
    def build(self, input_shape):
        self.coefficients = self.add_weight(shape=(self.degree), initializer='random', trainable=True)

    def call(self, input):
        y = (input * 0) + self.coefficients[0] #Quick and dirty way to have a matching size
        for i in range(self.degree-1):
            y += tf.pow(input, i+1) * self.coefficients[i+1]
        
        return y