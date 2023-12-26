#Shout outs, shout outs to: https://github.com/opconty/keras-shufflenetV2/blob/master/utils.py
import tensorflow as tf
import numpy as np
import InferredActivations.Inferrer as II
import Helpers as h
import logging
from keras import layers, models

logger = h.create_logger("shufflenetv2_dump.log")

TYPE = "control"
assert TYPE in ["control", "al", "pwlu"]
BOTTLENECK_RATIO = 0.5
SHUFFLE_BLOCKS = [3, 7, 3]
SCALE_FACTOR = 1.0

BATCH_SIZE = 128 #cifar-10 128/16, imagenette 16/4, imagenet2012 8/4
CONCAT_AUG = True
AUGMENT_FACTOR = 0.1

EPOCHS = 15
INIT_LRATE = 0.01
LRATE_SCHED = 6
LRATE_RATIO = 0.1
W_DECAY = 0.0
MOMENTUM = 0.9

IMAGE_SIZE = (224, 224)
print((
	'Beginning shuffle net v2 testing!'
	f'\n\t{TYPE} type'
	f'\n\t{BOTTLENECK_RATIO} bottleneck ratio'
	f'\n\t{SHUFFLE_BLOCKS} shuffle blocks'
	f'\n\t{SCALE_FACTOR} scale factor'
	f'\n\n\t{BATCH_SIZE} batch size'
	f'\n\t{AUGMENT_FACTOR} augment factor'
	f'\n\tConcatenate Augment? {CONCAT_AUG}'
	f'\n\n\t{EPOCHS} total epochs'
	f'\n\t{INIT_LRATE} initial learning rate'
	f'\n\tLearning rate schedule of {LRATE_RATIO} ratio every {LRATE_SCHED} epochs'
	f'\n\t{W_DECAY} weight decay'
	f'\n\t{MOMENTUM} SGD momentum'
))

activation_to_use = layers.Activation if TYPE == 'control' else (II.ActivationLinearizer if TYPE == "al" else II.PiecewiseLinearUnitV1)
activation_arg = "relu" if TYPE != "pwlu" else 5

print("\nPrepping CIFAR-10 Dataset")
train_ds, val_ds = h.load_cifar10(BATCH_SIZE)

train_ds = h.prepare_dataset(train_ds, IMAGE_SIZE, augment_factor=AUGMENT_FACTOR, concatenate_augment=CONCAT_AUG)
h.report_dataset_size("Training", train_ds, BATCH_SIZE)
val_ds = h.prepare_dataset(val_ds, IMAGE_SIZE)
h.report_dataset_size("Validation", val_ds, BATCH_SIZE)

print("\nDefining Shuffle Layers")
class ShuffleUnit(layers.Layer):
	def __init__(self, out_channels, strides=2, stage=1):
		super(ShuffleUnit, self).__init__()
		self.out_channels = out_channels
		self.strides = strides
		self.stage = stage

	def build(self, input_shape): #NOTE: Expecting Channels Last
		bottleneck_channels = int(self.out_channels * BOTTLENECK_RATIO)

		self.right = models.Sequential([
			layers.Conv2D(bottleneck_channels, 1, strides=1, padding='same'),
			layers.BatchNormalization(-1),
			activation_to_use(activation_arg),
			layers.DepthwiseConv2D(kernel_size=3, strides=self.strides, padding='same'),
			layers.BatchNormalization(-1),
			layers.Conv2D(bottleneck_channels, 1, strides=1, padding='same'),
			layers.BatchNormalization(-1),
			activation_to_use(activation_arg)
		])

		if self.strides >= 2:
			self.left_on_stride2 = models.Sequential([
				layers.DepthwiseConv2D(kernel_size=3, strides=2, padding='same'),
				layers.BatchNormalization(-1),
				layers.Conv2D(bottleneck_channels, 1, strides=1, padding='same'),
				layers.BatchNormalization(-1),
				activation_to_use(activation_arg)
			])

		#NOTE: I'm betting that this fails, here are the results after testing: Guess what, it DID! but for another reason than I thought lmao
		"""self.channel_shuffle = models.Sequential([
			layers.Reshape([height, width, 2, c_p_split]), #NOTE: I'm betting this fails
			layers.Permute((1, 2, 4, 3)), #Don't know why we switch from like (..., 2, 1) to (..., 1, 2) but I imagine it has to do with the reshape
			layers.Reshape([height, width, channels])
		])"""

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
			ShuffleUnit(self.channel_map[self.stage-1], strides=2, stage=self.stage),
			*[ShuffleUnit(self.channel_map[self.stage-1], strides=1, stage=self.stage) for _ in range(1, self.repeat+1)]
		])

		self.units.build(input_shape)

	def call(self, input):
		return self.units(input)

print(f"\nDefining ShuffleNetV2 Type: {TYPE}")
strat = tf.distribute.MirroredStrategy()
print(f"\t{strat.num_replicas_in_sync} Available Devices")

with strat.scope():
	optim = tf.keras.optimizers.experimental.SGD(INIT_LRATE, MOMENTUM, weight_decay=(W_DECAY if W_DECAY > 0 else None))

	#Mappings according to the bottleneck we have chosen
	assert BOTTLENECK_RATIO == 0.5 or BOTTLENECK_RATIO == 1 or BOTTLENECK_RATIO == 1.5 or BOTTLENECK_RATIO == 2
	out_dim_stage_two = {0.5:48, 1:116, 1.5:176, 2:244}

	out_channels_in_stage = 2 ** np.insert(np.arange(len(SHUFFLE_BLOCKS), dtype=np.float32), 0, 0) #ie [1, 1, 2, 4]
	out_channels_in_stage *= out_dim_stage_two[BOTTLENECK_RATIO]
	out_channels_in_stage[0] = 24 #according to the github first stage always has 24 output channels
	out_channels_in_stage *= SCALE_FACTOR
	out_channels_in_stage = out_channels_in_stage.astype(int)

	shufflenetv2 = models.Sequential([
		layers.Conv2D(24, 3, strides=2, padding='same', use_bias=False, input_shape=(*IMAGE_SIZE, 3)),
		activation_to_use(activation_arg),
		layers.MaxPool2D(3, 2),

		#ShuffleBlock(out_channels_in_stage, repeat=SHUFFLE_BLOCKS[0], stage=2),
		*[ShuffleBlock(out_channels_in_stage, repeat=SHUFFLE_BLOCKS[i], stage=i+2) for i in range(len(SHUFFLE_BLOCKS))],
		
		layers.Conv2D(1024 if BOTTLENECK_RATIO < 2 else 2048, 1, strides=1, padding='same'),
		activation_to_use(activation_arg),
		layers.GlobalMaxPooling2D(),
		layers.Dense(10, activation="softmax"),
	])

	metrics_to_use = [
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="T5"),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="T3"),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1, name="T1")
    ]

	calls = [
		tf.keras.callbacks.TerminateOnNaN(),
		tf.keras.callbacks.LearningRateScheduler(h.lr_schedule_creator(LRATE_SCHED, LRATE_RATIO))
	]

print("\nStarting training")
shufflenetv2.compile(optim, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=metrics_to_use)
hist = shufflenetv2.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=calls)

h.output_training_history(logger, hist)
h.output_validation_history(logger, hist)