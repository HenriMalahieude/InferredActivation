import tensorflow as tf
import InferredActivations.Inferrer as II
import Helpers as h
from keras import layers, models

logger = h.create_logger("mobilenext_dump.log")

#Man, if only there was a way to not have to copy and paste this.....
TYPE = "control"
assert TYPE in ["control", "pwlu", "al"]
AUGMENT_FACTOR = 0.05
CONCAT_AUG = True

BATCH_SIZE = 128 if TYPE == "control" else (64 if TYPE == "al" else 32)
EPOCHS = 15
INIT_LRATE = 0.05
#LRATE_SCHED = 5
#LRATE_RATIO = 0.1

W_DECAY = 0.00004
MOMENTUM = 0.9

IMAGE_SIZE = (225, 225)

print((
	'Beginning mobile neXt testing!'
	f'\n\t{TYPE} type'
	#f'\n\t{BOTTLENECK_RATIO} bottleneck ratio'
	#f'\n\t{SHUFFLE_BLOCKS} shuffle blocks'
	#f'\n\t{SCALE_FACTOR} scale factor'
	f'\n\n\t{BATCH_SIZE} batch size'
	f'\n\t{AUGMENT_FACTOR} augment factor'
	#f'\n\t{DROPOUT} of dropout'
	f'\n\tConcatenate Augment? {CONCAT_AUG}'
	f'\n\n\t{EPOCHS} total epochs'
	f'\n\t{INIT_LRATE} initial learning rate'
	f'\n\tLearning rate schedule of cosine ratio every -- epochs'
	f'\n\t{W_DECAY} weight decay'
	f'\n\t{MOMENTUM} SGD momentum'
))

act_to_use = layers.Activation if TYPE == 'control' else (II.ActivationLinearizer if TYPE == "al" else II.PiecewiseLinearUnitV1)
act_arg1 = "relu" if TYPE != "pwlu" else 5
act_arg2 = "relu6" if TYPE != "pwlu" else 5

print("\nPrepping CIFAR-10 Dataset")
train_ds, val_ds = h.load_cifar10(BATCH_SIZE)

train_ds = h.prepare_dataset(train_ds, IMAGE_SIZE, augment_factor=AUGMENT_FACTOR, concatenate_augment=CONCAT_AUG)
h.report_dataset_size("Training", train_ds, BATCH_SIZE)
val_ds = h.prepare_dataset(val_ds, IMAGE_SIZE)
h.report_dataset_size("Validation", val_ds, BATCH_SIZE)

print("\nDefining Sandglass Block")
class SandglassBlock(layers.Layer):
	def __init__(self, reduction_ratio, out_channels, stride=1):
		super(SandglassBlock, self).__init__()
		self.t = reduction_ratio
		self.n = out_channels
		self.s = stride

	def build(self, input_shape):
		m = input_shape[-1]

		self.sequence = models.Sequential([
			layers.DepthwiseConv2D(3, padding='same'),
			layers.BatchNormalization(-1),
			act_to_use(act_arg2),

			layers.Conv2D(m // self.t, 1),
			layers.BatchNormalization(-1),

			layers.Conv2D(self.n, 1),
			layers.BatchNormalization(-1),
			act_to_use(act_arg2),

			layers.DepthwiseConv2D(3, self.s, padding='same'),
			layers.BatchNormalization(-1),
		])

	def call(self, input):
		s1 = self.sequence(input)

		if input.shape.as_list() == s1.shape.as_list():
			return input + s1
		
		return s1
	
print("\nDefining Model")
strat = tf.distribute.MirroredStrategy()
print(f"\tUsing {strat.num_replicas_in_sync} GPUs")

with strat.scope():
	optim = tf.keras.optimizers.experimental.SGD(tf.keras.optimizers.schedules.CosineDecay(INIT_LRATE, 50000), MOMENTUM, weight_decay=(W_DECAY if W_DECAY > 0 else None))

	neXt = models.Sequential([
		layers.Conv2D(32, 3, strides=2, input_shape=(*IMAGE_SIZE, 3)),
		layers.BatchNormalization(-1),
		act_to_use(act_arg1),
		
		SandglassBlock(2, 96, 2),

		SandglassBlock(6, 144),

		SandglassBlock(6, 192, 2),
		SandglassBlock(6, 192),
		SandglassBlock(6, 192),

		SandglassBlock(6, 288, 2),
		SandglassBlock(6, 288),
		SandglassBlock(6, 288),

		*[SandglassBlock(6, 384) for _ in range(4)],

		SandglassBlock(6, 576, 2),
		*[SandglassBlock(6, 576) for _ in range(3)],

		SandglassBlock(6, 960),
		SandglassBlock(6, 960),

		SandglassBlock(6, 1280),

		layers.Conv2D(1280, 7),
		layers.BatchNormalization(-1),
		act_to_use(act_arg1),

		layers.Conv2D(10, 1),
		layers.Flatten(),
		layers.Activation('softmax') #Proper formatting for the thingy
	])
	
	metrics_to_use = [
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="T5"),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="T3"),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1, name="T1")
    ]

	calls = [
		tf.keras.callbacks.TerminateOnNaN(),
		#tf.keras.callbacks.LearningRateScheduler(h.lr_schedule_creator(LRATE_SCHED, LRATE_RATIO))
	]

#neXt.summary()
#neXt.layers[3].sequence.summary()

print(f"\nStarting {TYPE} training for MobileNeXt")
neXt.compile(optim, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=metrics_to_use)
hist = neXt.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=calls)

h.output_training_history(logger, hist)
h.output_validation_history(logger, hist)