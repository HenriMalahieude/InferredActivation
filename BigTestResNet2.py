import tensorflow as tf
import tensorflow_datasets as tfds
import logging
import InferredActivations.Inferrer as II

from keras import layers, models, datasets

#Setting up Logger
logger = logging.getLogger("internal_logger")
logger.setLevel(logging.DEBUG)
fileHandle = logging.FileHandler(filename="resnet_bigtest_dump.log")

formatter = logging.Formatter(fmt='%(message)s')
fileHandle.setFormatter(formatter)
logger.addHandler(fileHandle)

#Rest of the script
print("Starting ResNet50 Manual Sandbox")
DATASET = "cifar10" #cifar10 / imagenette / imagenet2012
BATCH_SIZE = 128 # cifar10 -> 128, imagenette -> 16, imagenet2012
AUGMENT = True
CONCAT = True
AUGMENT_FACTOR = 0.1
print("\t{} Dataset\n\t{} Batch Size\n\tAugmenting? {}\n\tConcatenating Augment? {}\n\t{} Augment Factor".format(DATASET, BATCH_SIZE, AUGMENT, CONCAT, AUGMENT_FACTOR))

EPOCHS = 50
INIT_LRATE = 0.01
LRATE_EPOCH_DEC = 5
LRATE_DECREASE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
print("\t{} Epochs\n\t{} Initial Learning Rate\n\t{} decrease per {} epochs\n\t{} EMA\n\t{} Weight Decay".format(EPOCHS, INIT_LRATE, LRATE_DECREASE, LRATE_EPOCH_DEC, MOMENTUM, WEIGHT_DECAY))

IMAGE_SIZE = (224, 224) #3 Color Channels is assumed
ACTIVATIONS = 'control' # 'control' / 'PLU' / 'AL'
print("\t{} Expected Image Size\n\t{} Activations".format(IMAGE_SIZE, ACTIVATIONS))

print("\nPreparing Dataset")
train_ds, test_ds = None, None
if DATASET == "cifar10":
	print("\tLoading CIFAR10")
	training, testing = datasets.cifar10.load_data()
	train_ds = tf.data.Dataset.from_tensor_slices(training).batch(BATCH_SIZE)
	test_ds = tf.data.Dataset.from_tensor_slices(testing).batch(BATCH_SIZE)
elif DATASET == "imagenette":
	print("\tLoading Imagenette")
	train_ds, test_ds = tfds.load("imagenette", split=["train", "validation"], as_supervised=True, batch_size=BATCH_SIZE)
else:
	"""
	Olga Russakovsky*, Jia Deng*, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, 
	Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg and Li Fei-Fei. (* = equal contribution) 
	ImageNet Large Scale Visual Recognition Challenge. arXiv:1409.0575, 2014.
	"""
	print("\tLoading ImageNet-2012")
	train_ds, test_ds = tfds.load("imagenet2012", split=["train", "validation"], as_supervised=True, batch_size=BATCH_SIZE)

print("\t\t{} Images Loaded for Training".format(train_ds.cardinality() * BATCH_SIZE))
print("\t\t{} Images Loaded for Validation/Testing".format(test_ds.cardinality() * BATCH_SIZE))

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
		if CONCAT:
			dsn = dsn.concatenate(ds.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False))
		else:
			dsn = ds.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

	print("\tDataset Prepared")
	return dsn.prefetch(buffer_size=tf.data.AUTOTUNE)

train_ds = prepare_dataset(train_ds, augment=AUGMENT)
test_ds = prepare_dataset(test_ds)

print("\nDefining ResNet Layer")
#Copied from DifferentModels/ResNet.py file
class ResidualBlock(layers.Layer):
    def __init__(self, 
                 filters = (64, 64, 256),
                 stride=1,
                 a_1 = layers.Activation("relu"),
                 a_2 = layers.Activation("relu"),
                 a_3 = layers.Activation("relu")):
        super(ResidualBlock, self).__init__()
        self.filters = filters
        self.stride = stride
        self.a1 = a_1
        self.a2 = a_2
        self.a3 = a_3

    def build(self, input_shape):
        f1, f2, f3 = self.filters

        self.plain = tf.keras.models.Sequential([
            layers.Conv2D(f1, 1, strides=self.stride),
            layers.BatchNormalization(),
            self.a1,

            layers.Conv2D(f2, 3, padding='same'),
            layers.BatchNormalization(),
            self.a2,

            layers.Conv2D(f3, 1),
            layers.BatchNormalization(),
        ])

        is_expanding = not (input_shape[-1] == f3)
        if is_expanding:
            self.id_pass = tf.keras.models.Sequential([
                layers.Conv2D(f3, 1, strides=self.stride),
                layers.BatchNormalization(),
            ])
        else:
            self.id_pass = layers.BatchNormalization()

        self.final_pass = self.a3
    
    def call(self, input):
        return self.final_pass(self.plain(input) + self.id_pass(input))

print("\nMulti-GPU Target for Training")
strat = tf.distribute.MirroredStrategy()
print("\tNumber of Devices: {}".format(strat.num_replicas_in_sync))

with strat.scope():
	custom_optim = tf.keras.optimizers.AdamW(INIT_LRATE, WEIGHT_DECAY, use_ema=(MOMENTUM > 0), ema_momentum=MOMENTUM)

	acts_al = {
		"a_1": II.ActivationLinearizer(),
		"a_2": II.ActivationLinearizer(),
		"a_3": II.ActivationLinearizer() 
	}

	acts_plu = {
		"a_1": II.PiecewiseLinearUnitV1(),
		"a_2": II.PiecewiseLinearUnitV1(),
		"a_3": II.PiecewiseLinearUnitV1(),
	}

	acts_to_use = {} if ACTIVATIONS == 'control' else (acts_plu if ACTIVATIONS == "PLU" else acts_al)

	res_net50 = models.Sequential([
		layers.Conv2D(64, 7, strides=2, input_shape=(*IMAGE_SIZE, 3)),
		layers.Activation('relu') if ACTIVATIONS == 'control' else (II.PiecewiseLinearUnitV1() if ACTIVATIONS == "PLU" else II.ActivationLinearizer()),
		layers.MaxPooling2D(3, 2),

		ResidualBlock((64, 64, 256), **acts_to_use),
		ResidualBlock((64, 64, 256), **acts_to_use),
		ResidualBlock((64, 64, 256), **acts_to_use),

		ResidualBlock((128, 128, 512), stride=2, **acts_to_use),
        ResidualBlock((128, 128, 512), **acts_to_use),
        ResidualBlock((128, 128, 512), **acts_to_use),

        ResidualBlock((256, 256, 1024), stride=2, **acts_to_use),
        ResidualBlock((256, 256, 1024), **acts_to_use),
        ResidualBlock((256, 256, 1024), **acts_to_use),
        ResidualBlock((256, 256, 1024), **acts_to_use),
        ResidualBlock((256, 256, 1024), **acts_to_use),
        ResidualBlock((256, 256, 1024), **acts_to_use),

        ResidualBlock((512, 512, 2048), stride=2, **acts_to_use),
        ResidualBlock((512, 512, 2048), **acts_to_use),
        ResidualBlock((512, 512, 2048), **acts_to_use),

		layers.AveragePooling2D(2, padding='same'),
		layers.Flatten(),

		layers.Dense(10 if (DATASET == "cifar10" or DATASET == "imagenette") else 1000, activation='softmax')
	])

	def learn_rate_scheduler(epoch, lr):
		interval_check = epoch % LRATE_EPOCH_DEC
		if interval_check == 0:
			return lr * min(max(LRATE_DECREASE, 0), 1)
		return lr

	calls = [
		tf.keras.callbacks.TerminateOnNaN(),
		tf.keras.callbacks.LearningRateScheduler(learn_rate_scheduler)
	]

	metrics_to_use = []
	if DATASET != "cifar10" and DATASET != "imagenette":
		metrics_to_use.extend([
			tf.keras.metrics.TopKCategoricalAccuracy(k=500, name="T500"),
			tf.keras.metrics.TopKCategoricalAccuracy(k=250, name="T250")
		])

	metrics_to_use.extend([
		tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="T5"),
		tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="T3"),
		tf.keras.metrics.TopKCategoricalAccuracy(k=1, name="T1"),
	])

print("\nTraining Model")
res_net50.compile(optimizer=custom_optim, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=metrics_to_use)
hist = res_net50.fit(train_ds, epochs=EPOCHS, validation_data=test_ds, callbacks=calls)

print("\n")
logger.info("{} {} Training History".format(DATASET, ACTIVATIONS))
if DATASET != 'cifar10' and DATASET != 'imagenette':
	logger.info("Training T500: {}".format(hist.history["T500"]))
	logger.info("Training T250: {}".format(hist.history["T250"]))
logger.info("Training T5: {}".format(hist.history["T5"]))
logger.info("Training T3: {}".format(hist.history["T3"]))
logger.info("Training T1: {}".format(hist.history["T1"]))
logger.info("Training Loss: {}".format(hist.history["loss"]))

print("\n")
if DATASET != 'cifar10' and DATASET != 'imagenette':
	logger.info("\nValidate T500: {}".format(hist.history["val_T500"]))
	logger.info("Validate T250: {}".format(hist.history["val_T250"]))
logger.info("Validate T5: {}".format(hist.history["val_T5"]))
logger.info("Validate T3: {}".format(hist.history["val_T3"]))
logger.info("Validate T1: {}".format(hist.history["val_T1"]))
logger.info("Validate Loss: {}".format(hist.history["val_loss"]))