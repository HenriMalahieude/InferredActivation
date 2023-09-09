import tensorflow as tf

#I didn't realize that keras.layers.Dense was actually the Linear thing
class Linear(tf.keras.layers.Layer): #Implementation of PyTorch Linear, hopefully keras handles the device stuff
    def __init__(self, 
                 out_features: int,
                 initializer: str = 'glorot_uniform',
                 bias: bool = True,
                ):
        super(Linear, self).__init__()
        self.out_features = out_features
        self.bias = bias
        self.initializer = initializer

    def build(self, input_shape):
        #NOTE: torch, why say y=xA^T + B ???? when you can simply say x*A??? I had to dig through so much documentation to understand what was going on
        #       I didn't even figure it out until someone said nn.Linear == tf.Dense
        self.A = self.add_weight(shape=(input_shape[-1], self.out_features), initializer = self.initializer, trainable = True)
        self.B = self.add_weight(shape=(1,), initializer = ('zeros' if not self.bias else self.initializer), trainable = self.bias)

    def call(self, input):
        y = tf.linalg.matmul(input, self.A) + self.B
        return y