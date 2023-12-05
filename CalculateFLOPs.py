import tensorflow as tf
import InferredActivations.Inferrer as II
from tensorflow.python.profiler import model_analyzer, option_builder
from keras import layers, models

class FireModule(layers.Layer):
    def __init__(self, squeeze=16, expand=64 , a1=II.ActivationLinearizer(), a2=II.ActivationLinearizer(), a3=II.ActivationLinearizer()):
        super(FireModule, self).__init__()
        self.squeeze=squeeze
        self.expand=expand

        self.activation1 = a1
        self.activation2 = a2
        self.activation3 = a3

    def build(self, input_shape):
        self.sLayer = models.Sequential([
            layers.Conv2D(self.squeeze, 1),
            self.activation1,
        ])

        self.eOneLayer = models.Sequential([
            layers.Conv2D(self.expand, 1),
            self.activation2,
        ])

        self.eThreeLayer = models.Sequential([
            layers.Conv2D(self.expand, 3, padding='same'),
            self.activation3,
        ])

        self.sLayer.build(input_shape)
        self.eOneLayer.build(self.sLayer.compute_output_shape(input_shape))
        self.eThreeLayer.build(self.sLayer.compute_output_shape(input_shape))

    def call(self, input):
        x = self.sLayer(input)

        left = self.eOneLayer(x)
        right = self.eThreeLayer(x)

        return layers.concatenate([left, right], axis=3)

model = models.Sequential([ #Standard, no bypasses
        layers.Conv2D(96, 7, strides=2, input_shape=(224, 224, 3)),
        layers.MaxPooling2D(3, 2),
        FireModule(),
        FireModule(),
        FireModule(squeeze=32, expand=128),
        layers.MaxPooling2D(3, 2),
        FireModule(squeeze=32, expand=128),
        FireModule(squeeze=48, expand=192),
        FireModule(squeeze=48, expand=192),
        FireModule(squeeze=64, expand=256),
        layers.MaxPooling2D(3, 2),
        FireModule(squeeze=64, expand=256),
        layers.Conv2D(1000, 1, strides=1),
        layers.AveragePooling2D(12, 1),
        II.ActivationLinearizer(),
        layers.Flatten(),
        layers.Dense(10, activation='softmax') #Because of differing class size
    ])

model.summary()

input_signature = [
    tf.TensorSpec(
        shape=(1, *params.shape[1:]), 
        dtype=params.dtype, 
        name=params.name
    ) for params in model.inputs
]

#Section copied from https://github.com/keras-team/tf-keras/issues/6
forward_graph = tf.function(model, input_signature).get_concrete_function().graph
options = option_builder.ProfileOptionBuilder.float_operation()
graph_info = model_analyzer.profile(forward_graph, options=options)
flops = graph_info.total_float_ops // 2
print(flops)