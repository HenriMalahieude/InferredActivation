import tensorflow as tf
import numpy as np
import InferredActivations.Inferrer as II
from tensorflow.python.profiler import model_analyzer, option_builder
from keras import layers, models

act_to_use = layers.Activation
act_arg1 = "relu"
act_arg2 = "relu6"

model = models.Sequential([
	
])

#Don't change what's below!
#------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------

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
flops = graph_info.total_float_ops // 2 #don't know why we're dividing by two, but I'm just following the example they gave me
print(flops)

#------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------