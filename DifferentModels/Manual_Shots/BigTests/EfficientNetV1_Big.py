import tensorflow as tf
import InferredActivations.Inferrer as II
import Helpers as h
import time
from keras import layers, models

logger = h.create_logger("efficient_net_v1_dump.log")

TYPE = "pwlu"
STATISTICS = True
assert TYPE in ["control", "pwlu", "al", "nupwlu"]
AUGMENT_FACTOR = 0.2
CONCAT_AUG = True

BATCH_SIZE = 128 if TYPE == "control" else (64 if TYPE == "al" else 32)
EPOCHS = 15
INIT_LRATE = 0.01
LRATE_SCHED = 30
LRATE_RATIO = 0.1

W_DECAY = 0.0
MOMENTUM = 0.9
DROPOUT = 0.2 #As states by paper
IMAGE_SIZE = (224, 224) #as stated by the paper

print((
	'Beginning efficient net v1-b0 testing!'
	f'\n\t{TYPE} type'
	f'\n\n\t{BATCH_SIZE} batch size'
	f'\n\t{AUGMENT_FACTOR} augment factor'
	f'\n\t{DROPOUT} of dropout'
	f'\n\tConcatenate Augment? {CONCAT_AUG}'
	f'\n\n\t{EPOCHS} total epochs + Statistical Analysis Period (PWLU only)? {STATISTICS}'
	f'\n\t{INIT_LRATE} initial learning rate'
	f'\n\tLearning rate schedule of {LRATE_RATIO} ratio every {LRATE_SCHED} epochs'
	f'\n\t{W_DECAY} weight decay'
	f'\n\t{MOMENTUM} SGD momentum'
))

pwlu_v = II.PiecewiseLinearUnitV1 if TYPE == "pwlu" else II.NonUniform_PiecewiseLinearUnit
act_to_use = layers.Activation if TYPE == 'control' else (II.ActivationLinearizer if TYPE == "al" else pwlu_v)
act_arg = "relu" if TYPE != "pwlu" and TYPE != "nupwlu" else 5
sig_arg = "sigmoid" if TYPE != "pwlu" and TYPE != "nupwlu" else 5

print("\nPrepping CIFAR-10 Dataset")
train_ds, val_ds = h.load_cifar10(BATCH_SIZE)

train_ds = h.prepare_dataset(train_ds, IMAGE_SIZE, augment_factor=AUGMENT_FACTOR, concatenate_augment=CONCAT_AUG)
h.report_dataset_size("Training", train_ds, BATCH_SIZE)
val_ds = h.prepare_dataset(val_ds, IMAGE_SIZE)
h.report_dataset_size("Validation", val_ds, BATCH_SIZE)

#Holy fuck thank god for https://github.com/GdoongMathew/EfficientNetV2/blob/master/efficientnetv2/model.py
print("\nDefining Efficient Net Layer")
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
				layers.Activation("relu"), #act_to_use(act_arg),
				layers.Conv2D(channels, 1, padding='same'),
				layers.Activation("sigmoid"), #act_to_use(sig_arg),
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

	def StatisticalAnalysisToggle(self, to):
		assert TYPE == "pwlu" or TYPE == "nupwlu"

		self.procedure1.layers[-1].StatisticalAnalysisToggle(to)
		if self.version == "normal":
			self.procedure2.layers[-1].StatisticalAnalysisToggle(to)

		if self.se_factor > 0:
			self.se_procedure.layers[3].StatisticalAnalysisToggle(to)
			self.se_procedure.layers[-1].StatisticalAnalysisToggle(to)

		if self.scale_factor == 1:
			self.procedure3.layers[2].StatisticalAnalysisToggle(to)
	
print(f"\nDefining {TYPE} Efficient Net V1-B0")
strat = tf.distribute.MirroredStrategy()
print(f"\tUsing {strat.num_replicas_in_sync} GPUs")

with strat.scope():
	optim = tf.keras.optimizers.experimental.SGD(INIT_LRATE, MOMENTUM, weight_decay=(W_DECAY if W_DECAY > 0 else None))

	effnetv1_b0 = models.Sequential([
		layers.Conv2D(32, 3, padding='same', input_shape=(*IMAGE_SIZE, 3)),  #224 x 224
		layers.BatchNormalization(-1),
		act_to_use(act_arg), 

		MBConv(1, 3, 16, stride=2),

		*[MBConv(6, 3, 24) for _ in range(2)],

		MBConv(6, 5, 40, stride=2),
		MBConv(6, 5, 40),

		MBConv(6, 3, 80, stride=2),
		*[MBConv(6, 3, 80) for _ in range(2)],

		MBConv(6, 5, 112, stride=2),
		*[MBConv(6, 5, 112) for _ in range(2)],

		*[MBConv(6, 5, 192) for _ in range(4)],

		MBConv(6, 3, 320, stride=2),

		layers.Conv2D(1280, 1),
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

#effnetv1_b0.summary()

print(f"\nStarting {TYPE} training")
effnetv1_b0.compile(optim, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=metrics_to_use)
if (TYPE == "pwlu" or TYPE == "nupwlu") and STATISTICS:
	h.PWLU_Set_StatisticalToggle(effnetv1_b0, "ShuffleBlock", True)
	print(f"\t\tBeginning Statistical Analysis")
	tme = time.time()
	for x, _ in train_ds:
		effnetv1_b0.call(x)
	print(f"\t\tStatistics took {time.time() - tme}s to calculate")
	h.PWLU_Set_StatisticalToggle(effnetv1_b0, "ShuffleBlock", False)
hist = effnetv1_b0.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=calls)

h.output_training_history(logger, hist)
h.output_validation_history(logger, hist)#"""