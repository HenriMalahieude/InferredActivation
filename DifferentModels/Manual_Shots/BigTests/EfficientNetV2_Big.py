#Locking this to the EfficientNetV2-S model found in: https://arxiv.org/pdf/2104.00298.pdf
import tensorflow as tf
import InferredActivations.Inferrer as II
import Helpers as h
from keras import layers, models

logger = h.create_logger("efficient_net_v2_dump.log")

TYPE = "control"
assert TYPE in ["control", "pwlu", "al"]
AUGMENT_FACTOR = 0.2
CONCAT_AUG = True

BATCH_SIZE = 64 if TYPE == "control" else (32 if TYPE == "al" else 16)
EPOCHS = 15
INIT_LRATE = 0.1
LRATE_SCHED = 5
LRATE_RATIO = 0.1

W_DECAY = 0.0
MOMENTUM = 0.9
DROPOUT = 0.2 #Halfway between min and max (see Table 6)
IMAGE_SIZE = (198, 198) #Exactly halfway between min and max image size that they used (Table 6)

print((
	'Beginning efficient net v2 testing!'
	f'\n\t{TYPE} type'
	#f'\n\t{BOTTLENECK_RATIO} bottleneck ratio'
	#f'\n\t{SHUFFLE_BLOCKS} shuffle blocks'
	#f'\n\t{SCALE_FACTOR} scale factor'
	f'\n\n\t{BATCH_SIZE} batch size'
	f'\n\t{AUGMENT_FACTOR} augment factor'
	f'\n\t{DROPOUT} of dropout'
	f'\n\tConcatenate Augment? {CONCAT_AUG}'
	f'\n\n\t{EPOCHS} total epochs'
	f'\n\t{INIT_LRATE} initial learning rate'
	f'\n\tLearning rate schedule of {LRATE_RATIO} ratio every {LRATE_SCHED} epochs'
	f'\n\t{W_DECAY} weight decay'
	f'\n\t{MOMENTUM} SGD momentum'
))

act_to_use = layers.Activation if TYPE == 'control' else (II.ActivationLinearizer if TYPE == "al" else II.PiecewiseLinearUnitV1)
act_arg = "relu" if TYPE != "pwlu" else 20
sig_arg = "sigmoid" if TYPE != 'pwlu' else 15

print("\nPrepping CIFAR-10 Dataset")
train_ds, val_ds = h.load_cifar10(BATCH_SIZE)

train_ds = h.prepare_dataset(train_ds, IMAGE_SIZE, augment_factor=AUGMENT_FACTOR, concatenate_augment=CONCAT_AUG)
h.report_dataset_size("Training", train_ds, BATCH_SIZE)
val_ds = h.prepare_dataset(val_ds, IMAGE_SIZE)
h.report_dataset_size("Validation", val_ds, BATCH_SIZE)

#Holy fuck thank god for https://github.com/GdoongMathew/EfficientNetV2/blob/master/efficientnetv2/model.py
print("\nDefining Efficient Net Layers")
class MBConv(layers.Layer):
	def __init__(self, scale_factor, kernel_size, output_channels, se_factor = 0, stride=1, version="normal"):
		super(MBConv, self).__init__()
		assert version in ["normal", "fused"]
		self.version = version
		self.scale_factor = scale_factor
		self.se_factor = se_factor
		self.kernel_size = kernel_size
		self.stride = stride
		self.output_channels = output_channels

	def build(self, input_shape):
		first_kern = 1 if self.version == "normal" else self.kernel_size
		first_stride = 1 if self.version == "normal" else self.stride

		last_kern = 1 if self.scale_factor != 1 else self.kernel_size
		last_stride = 1 if self.scale_factor != 1 else self.stride

		channels = int(input_shape[-1] * self.scale_factor)
		reduced_channels = int(input_shape[-1] * self.se_factor)

		self.final_move_check = (last_stride == first_stride == self.stride == 1) and input_shape[-1] == self.output_channels

		self.procedure1 = models.Sequential([
			layers.Conv2D(channels, first_kern, strides=first_stride, padding='same', use_bias=False),
			layers.BatchNormalization(-1), #NOTE: assuming channels last
			act_to_use(act_arg),
		])

		if self.version == "normal":
			self.procedure2 = models.Sequential([
				layers.DepthwiseConv2D(self.kernel_size, self.stride, padding="same", use_bias=False),
				layers.BatchNormalization(-1),
				act_to_use(act_arg),
			])

		self.drop1 = layers.Dropout(DROPOUT)

		if self.se_factor > 0:
			self.se_procedure = models.Sequential([
				layers.GlobalAveragePooling2D(),
				layers.Reshape((1, 1, channels)),
				layers.Conv2D(reduced_channels, 1, padding='same'),
				act_to_use(act_arg),
				layers.Conv2D(channels, 1, padding='same'),
				act_to_use(sig_arg),
			])

		self.procedure3 = models.Sequential([
			layers.Conv2D(self.output_channels, last_kern, strides = last_stride, padding='same', use_bias=False),
			layers.BatchNormalization(-1),
		])

		if self.scale_factor == 1:
			self.procedure3.add(act_to_use(act_arg))
		
		if self.final_move_check:
			self.procedure3.add(layers.Dropout(DROPOUT, noise_shape=(None, 1, 1, 1)))

	def call(self, input):
		s1 = self.procedure1(input)

		if self.version == "normal":
			s1 = self.procedure2(s1)
		
		s1 = self.drop1(s1)

		if self.se_factor > 0:
			s2 = self.se_procedure(s1)
			s1 = s1 * s2

		s3 = self.procedure3(s1)

		if self.final_move_check:
			return s3 + input
		
		return s3


print(f"\nDefining Model: {TYPE}")
strat = tf.distribute.MirroredStrategy()
print(f"\tUsing {strat.num_replicas_in_sync} GPUs")

with strat.scope():
	optim = tf.keras.optimizers.experimental.SGD(INIT_LRATE, MOMENTUM, weight_decay=(W_DECAY if W_DECAY > 0 else None))

	efficient_net_v2_s = models.Sequential([
		layers.Conv2D(24, 3, strides=2, input_shape=(*IMAGE_SIZE, 3)),
		layers.BatchNormalization(-1),
		act_to_use(act_arg),
		*[MBConv(1, 3, 24, version="fused") for _ in range(2)], #4r

		MBConv(4, 3, 48, stride=2, version='fused'),            #1r
		*[MBConv(4, 3, 48, version="fused") for _ in range(3)], #3r

		MBConv(4, 3, 64, stride=2, version="fused"),            #1r
		*[MBConv(4, 3, 64, version="fused") for _ in range(3)], #3r

		MBConv(4, 3, 128, se_factor=0.25, stride=2),            #(3r + 1s)
		*[MBConv(4, 3, 128, se_factor=0.25) for _ in range(5)], #(15r + 5s)

		*[MBConv(6, 3, 160, se_factor=0.25) for _ in range(9)],	#(27r + 9s)

		MBConv(6, 3, 256, se_factor=0.25, stride=2),            #(3r + 1s)
		*[MBConv(6, 3, 256, se_factor=0.25) for _ in range(14)],#(3r + 1s) * 14

		layers.Conv2D(1280, 1, padding='same', use_bias=False),
		layers.BatchNormalization(-1),
		act_to_use(act_arg),

		layers.Dropout(DROPOUT),

		layers.GlobalAveragePooling2D(),
		layers.Dense(10, activation="softmax")
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

#efficient_net_v2_s.summary()

print(f"\nStarting {TYPE} training")
efficient_net_v2_s.compile(optim, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=metrics_to_use)
hist = efficient_net_v2_s.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=calls)

h.output_training_history(logger, hist)
h.output_validation_history(logger, hist)#"""