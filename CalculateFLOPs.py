import tensorflow as tf
import numpy as np
import InferredActivations.Inferrer as II
from tensorflow.python.profiler import model_analyzer, option_builder
from keras import layers, models

class ShuffleUnit(layers.Layer):
	def __init__(self, out_channels, strides=2, stage=1):
		super(ShuffleUnit, self).__init__()
		self.out_channels = out_channels
		self.strides = strides
		self.stage = stage

	def build(self, input_shape): #NOTE: Expecting Channels Last
		bottleneck_channels = int(self.out_channels * 0.5)

		self.right = models.Sequential([
			layers.Conv2D(bottleneck_channels, 1, strides=1, padding='same'),
			layers.BatchNormalization(-1),
			layers.Activation('relu'),
			layers.DepthwiseConv2D(kernel_size=3, strides=self.strides, padding='same'),
			layers.BatchNormalization(-1),
			layers.Conv2D(bottleneck_channels, 1, strides=1, padding='same'),
			layers.BatchNormalization(-1),
			layers.Activation('relu')
		])

		if self.strides >= 2:
			self.left_on_stride2 = models.Sequential([
				layers.DepthwiseConv2D(kernel_size=3, strides=2, padding='same'),
				layers.BatchNormalization(-1),
				layers.Conv2D(bottleneck_channels, 1, strides=1, padding='same'),
				layers.BatchNormalization(-1),
				layers.Activation('relu')
			])

	#Expects channels last, returns c_hat, c
	def channel_split(self, input):
		channels = input.shape.as_list()[-1]
		cp_split = channels // 2
		return input[:, :, :, 0:cp_split], input[:, :, :, cp_split:]
	
	def channel_shuffle(self, input):
		height, width, channels = input.shape.as_list()[-3:]
		cp_split = channels // 2
		x = layers.Reshape([height, width, 2, cp_split])(input)
		x = layers.Permute((1,2,4,3))(x)
		return layers.Reshape([height, width, channels])(x)



	def call(self, input):
		if self.strides < 2:
			c_hat, c = self.channel_split(input)
			input = c

		r1 = self.right(input)

		if self.strides < 2:
			ret = layers.Concatenate(-1)([r1, c_hat])
		else:
			l1 = self.left_on_stride2(input)
			ret = layers.Concatenate(-1)([r1, l1])

		return self.channel_shuffle(ret)

class ShuffleBlock(layers.Layer):
	def __init__(self, channel_map, repeat=1, stage=1):
		self.channel_map = channel_map
		self.repeat = repeat
		self.stage = stage
		super(ShuffleBlock, self).__init__()

	def build(self, input_shape):
		self.units = models.Sequential([
			ShuffleUnit(self.channel_map[self.stage-1], strides=2, stage=self.stage), # +3
			*[ShuffleUnit(self.channel_map[self.stage-1], strides=1, stage=self.stage) for _ in range(1, self.repeat+1)] # +2 * repeat
		])

		self.units.build(input_shape)

	def call(self, input):
		return self.units(input)

out_channels_in_stage = 2 ** np.insert(np.arange(len([3, 7, 3]), dtype=np.float32), 0, 0) #ie [1, 1, 2, 4]
out_channels_in_stage *= 48
out_channels_in_stage[0] = 24 #according to the github first stage always has 24 output channels
out_channels_in_stage = out_channels_in_stage.astype(int)

model = models.Sequential([
    layers.Conv2D(24, 3, strides=2, padding='same', use_bias=False, input_shape=(224, 224, 3)),
    layers.Activation('relu'),
    layers.MaxPool2D(3, 2),

    #ShuffleBlock(out_channels_in_stage, repeat=SHUFFLE_BLOCKS[0], stage=2),
    *[ShuffleBlock(out_channels_in_stage, repeat=[3, 7, 3][i], stage=i+2) for i in range(len([3, 7, 3]))],
    
    layers.Conv2D(1024, 1, strides=1, padding='same'),
    layers.Activation('relu'),
    layers.GlobalMaxPooling2D(),
    layers.Dense(10, activation="softmax"),
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