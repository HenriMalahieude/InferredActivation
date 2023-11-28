import tensorflow as tf
import tensorflow_datasets as tfds
import logging
import InferredActivations.Inferrer as II

from keras import layers, models

print("Flip Flocker Idea by Dr. Brisk")

#Hyperparameters
BATCH_SZ = 128
AUGMENT = False
CONCAT_AUG = False
AUGMENT_FACTOR = 0.05
print("\t{} Batching\n\tAugmenting? {}\n\tConcatenating Augment? {}\n\t{} Augment Factor".format(BATCH_SZ, AUGMENT, CONCAT_AUG, AUGMENT_FACTOR))

UNLOCKED_EPOCH = 2
EPOCH_TOTAL = 5
INIT_LRATE = 0.001
LRATE_SCHEDULE = 500
LRATE_RATIO = 0.1
MOMENTUM = 0.0
W_DECAY = 0.0
print("\t{} Total Epochs\n\t{} Initial Learning Rate\n\t{} Ratio Reduction per {} Epoch\n\t{} EMA\n\t{} Weight Decay".format(
	EPOCH_TOTAL, INIT_LRATE, LRATE_RATIO, LRATE_SCHEDULE, MOMENTUM, W_DECAY
))

MODEL_T = "lenet"
assert MODEL_T in ["lenet", "alexnet-c", "alexnet-i"] #loads mnist, cifar10, or imagenette

DROPOUT = 0.5
TEST = "control"
assert TEST in ["control", "flipper", "RPWLU", "RAL"] #basic, special trainer, Normal PWLU, Normal AL
print("\t{} Model in {} Test Configuration\n\t{} Dropout (if used)".format(MODEL_T, TEST, DROPOUT))

IMAGE_SIZE = None #Will be autoset, so don't touch this

print("\nPreparing Dataset")
train_ds, val_ds = None, None
if MODEL_T == "lenet":
	print("\tLoading MNIST")
	IMAGE_SIZE = (32, 32)
	(x_train, y_train), (x_val, y_val) = tf.keras.datasets.mnist.load_data()

	x_train = tf.pad(x_train, [[0, 0], [2,2], [2,2]]) #Padding it up to 32 x 32
	x_val = tf.pad(x_val, [[0, 0], [2,2], [2,2]])

	train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(BATCH_SZ)
	val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(BATCH_SZ)
elif MODEL_T == "alexnet-c":
	print("\tLoading CIFAR-10")
	IMAGE_SIZE = (227, 227)
	training, testing = tf.keras.datasets.cifar10.load_data()
	train_ds = tf.data.Dataset.from_tensor_slices(training).batch(BATCH_SZ)
	val_ds = tf.data.Dataset.from_tensor_slices(testing).batch(BATCH_SZ)
elif MODEL_T == "alexnet-i":
	print("\tLoading Imagenette")
	IMAGE_SIZE = (227, 227)
	train_ds, val_ds = tfds.load("imagenette", split=["train", "validation"], as_supervised=True, batch_size=BATCH_SZ)

print("\t\t{} Images Loaded for Training".format(train_ds.cardinality() * BATCH_SZ))
print("\t\t{} Images Loaded for Validation/Testing".format(val_ds.cardinality() * BATCH_SZ))

resize_and_rescale = models.Sequential([
	layers.Resizing(*IMAGE_SIZE) if MODEL_T != "lenet" else layers.Reshape((*IMAGE_SIZE, 1)),
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

for x, y in train_ds:
	print(x.shape, y.shape)
	break

print("\nBuilding {}".format(MODEL_T))
strat = tf.distribute.MirroredStrategy()
print("\t{} Available Training Devices".format(strat.num_replicas_in_sync))

with strat.scope():
	custom_optimizer = tf.keras.optimizers.AdamW(INIT_LRATE, W_DECAY, use_ema=(MOMENTUM > 0), ema_momentum=MOMENTUM)

	def learn_rate_scheduler(epoch, lr):
		interval_check = epoch % LRATE_SCHEDULE
		if interval_check == 0:
			return lr * min(max(LRATE_RATIO, 0), 1)
		return lr

	calls = [
		tf.keras.callbacks.TerminateOnNaN(),
		tf.keras.callbacks.LearningRateScheduler(learn_rate_scheduler)
	]

	metrics_to_use = [
		tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="T5"),
		tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="T3"),
		tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1, name="T1"),
	]

	neural_net = None
	replacement = II.ActivationLinearizer if TEST != "RPLU" else II.PiecewiseLinearUnitV1
	if MODEL_T == "lenet":
		t_args = ["tanh"] if TEST != "RPLU" else []
		s_args = ["sigmoid"] if TEST != "RPLU" else []

		neural_net = models.Sequential([
			layers.Conv2D(6, 5, input_shape=(*IMAGE_SIZE, 1)),
			layers.Activation('tanh') if TEST == "control" else replacement(*t_args),
			layers.AveragePooling2D(2),
			layers.Activation('sigmoid') if TEST == "control" else replacement(*s_args),

			layers.Conv2D(16, 5),
			layers.Activation('tanh') if TEST == "control" else replacement(*t_args),
			layers.AveragePooling2D(2),
			layers.Activation('sigmoid') if TEST == "control" else replacement(*s_args),

			layers.Conv2D(120, 5),
			layers.Activation('tanh') if TEST == "control" else replacement(*t_args),
			layers.Flatten(),

			layers.Dense(84),
			layers.Activation('tanh') if TEST == "control" else replacement(*t_args),
			layers.Dense(10, activation='softmax'),
		])
	elif MODEL_T.startswith("alexnet"):
		inits = {
			"zero-gaussian" : tf.keras.initializers.RandomNormal(0.0, 0.01, seed=15023),
			"const1": tf.keras.initializers.Constant(1),
			"const0": tf.keras.initializers.Constant(0),
		}
		
		kwargs = [
			{"kernel_initializer": inits["zero-gaussian"], "bias_initializer": inits["const0"]},
			{"kernel_initializer": inits["zero-gaussian"], "bias_initializer": inits["const1"]},
		]

		neural_net = models.Sequential([
			layers.Conv2D(96, 11, strides=(4, 4), input_shape=(*IMAGE_SIZE, 3), **kwargs[0]),
			layers.Activation('relu') if TEST == "control" else replacement(),
			layers.MaxPooling2D(pool_size=(3, 3), strides=(2,2)),

			layers.Conv2D(256, 5, padding='same', **kwargs[1]),
			layers.Activation('relu') if TEST == "control" else replacement(),
			layers.MaxPooling2D(pool_size=(3,3), strides=(2,2)),

			layers.Conv2D(384, 3, padding='same', **kwargs[1]),
			layers.Activation('relu') if TEST == "control" else replacement(),
			layers.Conv2D(384, 3, padding='same', **kwargs[1]),
			layers.Activation('relu') if TEST == "control" else replacement(),
			layers.Conv2D(256, 3, padding='same', **kwargs[0]),
			layers.Activation('relu') if TEST == "control" else replacement(),
			layers.MaxPooling2D(pool_size=(3, 3), strides=(2,2)),

			layers.Flatten(),
			layers.Dense(4096, **kwargs[0]),
			layers.Activation('relu') if TEST == "control" else replacement(),
			layers.Dropout(DROPOUT),

			layers.Dense(4096, **kwargs[0]),
			layers.Activation('relu') if TEST == "control" else replacement(),
			layers.Dropout(DROPOUT),

			layers.Dense(10, **kwargs[0]),
			layers.Activation('softmax')
		])

#neural_net.summary(line_length=80)

print("\nReadying Training for {} {} variation".format(MODEL_T, TEST))

if TEST in ["control", "RPWLU", "RAL"]:
	print("\tNormal Training")
	neural_net.compile(custom_optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=metrics_to_use)
	neural_net.fit(train_ds, epochs=EPOCH_TOTAL, validation_data=val_ds, callbacks=calls)
elif TEST == "flipper": #Can only be completed with AL since PWLU doesn't have the possibility..... OR I could hack that in?
	print("\tFlip Flocking Ahead, starting unlocked training")
	neural_net.compile(custom_optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=metrics_to_use)
	neural_net.fit(train_ds, epochs=UNLOCKED_EPOCH, validation_data=val_ds, callbacks=calls)

	#lock
	print("\tLocking Layers")
	for lyr in neural_net.layers:
		if lyr.name.startswith("activation_linearizer"):
			lyr.lock_boundaries(force_to=True)
			lyr.extract_linears() #Checking things
	
	#NOTE: unsure if I need to recompile this stuff
	#neural_net.compile(custom_optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=metrics_to_use)
	neural_net.fit(train_ds, epochs=(EPOCH_TOTAL-UNLOCKED_EPOCH), validation_data=val_ds, callbacks=calls)
	
	print("\tFinished, ensuring that boundaries were not trained")
	for lyr in neural_net.layers:
		if lyr.name.startswith("activation_linearizer"):
			lyr.extract_linears()