import tensorflow as tf
import tensorflow_datasets as tfds
import InferredActivations.Inferrer as II
import logging

from keras import layers, models

MODEL_T = "PWLU" #control vs PWLU vs AL
DROPOUT = 0.5

DATASET = "imagenette" #imagenette vs cifar-10 vs imagenet2012
BATCH_SIZE = 4 #cifar-10 128/16, imagenette 16/4, imagenet2012 8/4
AUGMENT = True
CONCAT_AUG = True
AUGMENT_FACTOR = 0.11

EPOCHS = 20
INIT_LRATE = 0.04 #0.04 according to paper github
LRATE_SCHED = 5 #Could not locate the schedule they used
LRATE_RATIO = 0.1#`                                     `
W_DECAY = 0.0002 #0.0002 according to paper github
MOMENTUM = 0.9 #assuming SGD here (0.9 according to paper github)

IMAGE_SIZE = (224, 224)

assert MODEL_T in ["control", "PWLU", "AL"]
assert DATASET in ["imagenette", "cifar-10", "imagenet2012"]
print("Beginning SqueezeNet (Fire Module) Big Sandbox")
print(('\t{} Model'
	  '\n\t\tw/ {} Dropout'
	  '\n\t{} Dataset'
	  '\n\t\t{} Batch Size'
	  '\n\t\tAugmenting? {}'
	  '\n\t\tConcatenating Augment? {}'
	  '\n\t\t{} Augment Factor'
	  '\n\t{} Epochs'
	  '\n\t{} Initial Learning Rate'
	  '\n\t\tw/ {} Ratio every {} Epochs'
	  '\n\t{} Weight Decay'
	  '\n\t{} SGD Momentum').format(MODEL_T, DROPOUT, DATASET, BATCH_SIZE, AUGMENT, CONCAT_AUG, AUGMENT_FACTOR, EPOCHS, INIT_LRATE, LRATE_RATIO, LRATE_SCHED, W_DECAY, MOMENTUM))

#Setting up Logger
print("\nSetting up Logger")
logger = logging.getLogger("internal_logger")
logger.setLevel(logging.DEBUG)
fileHandle = logging.FileHandler(filename="squeeze_bigtest_dump.log")

formatter = logging.Formatter(fmt='%(message)s')
fileHandle.setFormatter(formatter)
logger.addHandler(fileHandle)

print("\nPrepping {} Dataset".format(DATASET))
train_ds, val_ds = None, None
if DATASET == "cifar-10":
	training, validation = tf.keras.datasets.cifar10.load_data()
	train_ds = tf.data.Dataset.from_tensor_slices(training).batch(BATCH_SIZE)
	val_ds = tf.data.Dataset.from_tensor_slices(validation).batch(BATCH_SIZE)
elif DATASET == "imagenette":
	train_ds, val_ds = tfds.load("imagenette", split=["train", "validation"], as_supervised=True, batch_size=BATCH_SIZE)
elif DATASET == "imagenet2012":
	train_ds, val_ds = tfds.load("imagenet2012", split=["train", "validation"], as_supervised=True, batch_size=BATCH_SIZE)

resize_and_rescale = models.Sequential([
	layers.Resizing(*IMAGE_SIZE),
	layers.Rescaling(1./255),
])

data_augmentation = models.Sequential([
	layers.RandomFlip(),
	layers.RandomContrast(AUGMENT_FACTOR),
	layers.RandomBrightness(AUGMENT_FACTOR),
	layers.RandomRotation(AUGMENT_FACTOR),
	resize_and_rescale
])

def prepare_dataset(ds, augment=False):
	dsn = ds.map((lambda x, y: (resize_and_rescale(x, training=True), y)), num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

	if augment:
		augmenter = (lambda x, y: (data_augmentation(x, training=True), y))
		if CONCAT_AUG:
			dsn = dsn.concatenate(ds.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False))
		else:
			dsn = ds.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

	print("\tDataset Prepared")
	return dsn.prefetch(buffer_size=tf.data.AUTOTUNE)

train_ds = prepare_dataset(train_ds, augment=AUGMENT)
val_ds = prepare_dataset(val_ds)

print("\nDefining the Fire Module")
class FireModule(layers.Layer):
    def __init__(self, squeeze=16, expand=64 , a1=layers.Activation('relu'), a2=layers.Activation('relu'), a3=layers.Activation('relu')):
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
    
print("\nEstablishing Multi-GPU Training Target")
strat = tf.distribute.MirroredStrategy()
print("\tDevices to train on: {}".format(strat.num_replicas_in_sync))

with strat.scope():
	optim = tf.keras.optimizers.experimental.SGD(INIT_LRATE, MOMENTUM, weight_decay=(W_DECAY if W_DECAY > 0 else None))
	replacement = layers.Activation if MODEL_T == "control" else (II.ActivationLinearizer if MODEL_T == "AL" else II.PiecewiseLinearUnitV1)
	r_args = ["relu"] if MODEL_T == "control" else []

	def produceFireModule(s=16, e=64):
		return FireModule(
			s,
			e,
			replacement(*r_args),
			replacement(*r_args),
			replacement(*r_args)
		)
	
	squeeze_net = models.Sequential([
     	layers.Conv2D(96, 7, strides=2, input_shape=(*IMAGE_SIZE, 3)),
        layers.MaxPooling2D(3, 2),
        produceFireModule(16, 64),
        produceFireModule(16, 64),
        produceFireModule(32, 128),
        layers.MaxPooling2D(3, 2),
        produceFireModule(32, 128),
        produceFireModule(48, 192),
        produceFireModule(48, 192),
        produceFireModule(64, 256),
        layers.MaxPooling2D(3, 2),
        produceFireModule(64, 256),
        layers.Conv2D((1000 if DATASET == "imagenet2012" else 10), 1, strides=1),
        layers.AveragePooling2D(12, 1),
        layers.Flatten(),
        layers.Activation("softmax")
	])

	def learn_rate_scheduler(epoch, lr):
		interval_check = epoch % LRATE_SCHED
		if interval_check == 0:
			return lr * min(max(LRATE_RATIO, 0), 1)
		return lr

	calls = [
		tf.keras.callbacks.TerminateOnNaN(),
		tf.keras.callbacks.LearningRateScheduler(learn_rate_scheduler)
	]

	metrics_to_use = []
	if DATASET != "cifar-10" and DATASET != "imagenette":
		metrics_to_use.extend([
			tf.keras.metrics.SparseTopKCategoricalAccuracy(k=500, name="T500"),
			tf.keras.metrics.SparseTopKCategoricalAccuracy(k=250, name="T250")
		])

	metrics_to_use.extend([
		tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="T5"),
		tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="T3"),
		tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1, name="T1"),
	])

#squeeze_net.summary()

print("\nTraining Squeeze Net in {} {} conditions".format(MODEL_T, DATASET))
squeeze_net.compile(optim, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=metrics_to_use)
hist = squeeze_net.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=calls)

logger.info("\n")
logger.info("{} {} Training History".format(DATASET, MODEL_T))
if DATASET != 'cifar-10' and DATASET != 'imagenette':
	logger.info("Training T500: {}".format(hist.history["T500"]))
	logger.info("Training T250: {}".format(hist.history["T250"]))
logger.info("Training T5: {}".format(hist.history["T5"]))
logger.info("Training T3: {}".format(hist.history["T3"]))
logger.info("Training T1: {}".format(hist.history["T1"]))
logger.info("Training Loss: {}".format(hist.history["loss"]))

logger.info("\n")
if DATASET != 'cifar-10' and DATASET != 'imagenette':
	logger.info("Validate T500: {}".format(hist.history["val_T500"]))
	logger.info("Validate T250: {}".format(hist.history["val_T250"]))
logger.info("Validate T5: {}".format(hist.history["val_T5"]))
logger.info("Validate T3: {}".format(hist.history["val_T3"]))
logger.info("Validate T1: {}".format(hist.history["val_T1"]))
logger.info("Validate Loss: {}".format(hist.history["val_loss"]))