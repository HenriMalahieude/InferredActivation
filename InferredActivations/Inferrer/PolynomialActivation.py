import tensorflow as tf
from keras import layers

#From https://arxiv.org/pdf/1906.09529.pdf
class SLAF_PolynomialActivation(layers.Layer):
    def __init__(self, degree=5, bounds=True, max_val=1000, min_val=-1000):
        super(SLAF_PolynomialActivation, self).__init__()
        assert min_val < max_val
        
        self.degree = max(degree, 2)
        self.max_val = min(max_val, 1000)
        self.min_val = max(min_val, -1000)
        self.bounds = bounds

    
    def build(self, input_shape):
        self.coefficients = self.add_weight(shape=(self.degree,), initializer='random_uniform', trainable=True)

    def call(self, input):
        if self.bounds: #Added in after testing to avoid 'nan' loss. Seems to be a case of the exploding gradients
            bounding = tf.cast(tf.math.logical_and(tf.math.greater(input, self.min_val), tf.math.less(input, self.max_val)), input.dtype)
            input *= bounding

        y = 0
        for i in range(self.degree):
            y += tf.pow(input, i) * self.coefficients[i]

        return y