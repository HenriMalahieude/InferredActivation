import tensorflow as tf
import tensorflow_datasets as tfds
import logging
import InferredActivations.Inferrer as II

from keras import layers, models

#Setting up Logger
logger = logging.getLogger("internal_logger")
logger.setLevel(logging.DEBUG)
fileHandle = logging.FileHandler(filename="se_net_bigtest_dump.log")

formatter = logging.Formatter(fmt='%(message)s')
fileHandle.setFormatter(formatter)
logger.addHandler(fileHandle)

print("Beginning Squeeze/Excitation Network Sandbox 2")

DATASET = "imagenette"
assert DATASET in ["imagenette", "cifar-10", "imagenet2012"]
AUGMENT_DATA = True
CONCAT_AUG = True
AUGMENT_FACTOR = 0.1

print("\t{} dataset\n\t\tAugmenting? {}\n\t\tConcatenating Augment? {}\n\t\t{} Augment Factor".format(DATASET, AUGMENT_DATA, CONCAT_AUG, AUGMENT_FACTOR))

MODEL_T = "control"
assert MODEL_T in ["control", "PWLU", "AL"]
REDUCTION_RATIO = 16 #16 as paper says

print("\t{} version\n\t\tw/ {} Reduction Ratio".format(MODEL_T, REDUCTION_RATIO))

BATCH_SZ = 8 #cifar-10 128, imagenette 8, imagenet2012 16
MAX_EPOCH = 20

#Using Learning Rate Scheduler
INIT_LRATE = 0.01 #0.6 initial by paper
LRATE_SCHED = 30 #30 epochs by paper
LRATE_RATIO = 0.1 #0.1 ratio by paper

#Using SGD Experimental Optimizer
MOMENTUM = 0.9 #0.9 by paper
W_DECAY = 0 #0 by paper

IMAGE_SIZE = (224, 224) #Assumed Channels last

print("""\t{} Batch Size
\t{} Epochs
\t{} Initial Learning Rate
\t{} Decrease Ratio Per {} Epochs
\t{} SGD momentum
\t{} Image Size""".format(BATCH_SZ, MAX_EPOCH, INIT_LRATE, LRATE_RATIO, LRATE_SCHED, MOMENTUM, IMAGE_SIZE))

print("\nPrepping {} Dataset".format(DATASET))
train_ds, val_ds = None, None
if DATASET == "imagenet2012":
	print("\tLoading Imagenet2012")
	train_ds, val_ds = tfds.load("imagenet2012", split=["train", "validation"], as_supervised=True, batch_size=BATCH_SZ)
elif DATASET == "imagenette":
	print("\tLoading Imagenette")
	train_ds, val_ds = tfds.load("imagenette", split=["train", "validation"], as_supervised=True, batch_size=BATCH_SZ)
elif DATASET == "cifar-10":
	print("\tLoading CIFAR-10")
	training, testing = tf.keras.datasets.cifar10.load_data()
	train_ds = tf.data.Dataset.from_tensor_slices(training).batch(BATCH_SZ)
	val_ds = tf.data.Dataset.from_tensor_slices(testing).batch(BATCH_SZ)

print("\t\t{} Images Loaded for Training".format(train_ds.cardinality() * BATCH_SZ))
print("\t\t{} Images Loaded for Validation/Testing".format(val_ds.cardinality() * BATCH_SZ))

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

train_ds = prepare_dataset(train_ds, augment=AUGMENT_DATA)
val_ds = prepare_dataset(val_ds)

print("\nDefining Layers")
#Taken from https://github.com/taki0112/SENet-Tensorflow/tree/master as source
class Squeeze_Excitation_Block(layers.Layer):
    def __init__(self, ratio,
                 a1 = layers.Activation('relu'),
                 a2 = layers.Activation('sigmoid')):
        super(Squeeze_Excitation_Block, self).__init__()
        self.ratio = ratio
        self.activation1 = a1
        self.activation2 = a2
    
    def build(self, input_shape):
        print(input_shape)
        self.excitation = models.Sequential([
            layers.GlobalAveragePooling2D(input_shape=input_shape), #This entire thing doesn't work btw, something to do WITH INPUT SHAPES BEING WRONG ALL THE TIME GODDAMN IT

            layers.Dense(units=(input_shape[-1]/self.ratio), use_bias=False), #Rest of this is the "excitation layer"
            self.activation1,
            layers.Dense(units=input_shape[-1], use_bias=False),
            self.activation2,
            layers.Reshape([1, 1, input_shape[-1]]) #So we can multiply to the input
        ])
    
    def call(self, input):
        return input * self.excitation(input) #Otherwise known as "The Scale"

class SE_ResidualBlock(layers.Layer):
    def __init__(self, 
                 filters=(64,64,256),
                 stride=1,

                 #Residual Activations
                 a1 = layers.Activation("relu"),
                 a2 = layers.Activation("relu"),
                 a3 = layers.Activation("relu"),
                 
                 #Squeeze-Excitation Activations
                 ase1 = layers.Activation('relu'),
                 ase2 = layers.Activation('sigmoid')):
        super(SE_ResidualBlock, self).__init__()
        self.filters = filters
        self.stride = stride
        self.activation1 = a1
        self.activation2 = a2
        self.activation3 = a3
        self.activation_se1 = ase1
        self.activation_se2 = ase2
    
    def build(self, input_shape):
        f1, f2, f3 = self.filters

        self.normal_pass = models.Sequential([
            layers.Conv2D(f1, 1, strides=self.stride),
            layers.BatchNormalization(),
            self.activation1,

            layers.Conv2D(f2, 3, padding='same'),
            layers.BatchNormalization(),
            self.activation2,

            layers.Conv2D(f3, 1),
            layers.BatchNormalization(),
        ])

        self.excitation = models.Sequential([
            layers.GlobalAveragePooling2D(),

            layers.Dense(units=(f3/REDUCTION_RATIO), use_bias=False), #Rest of this is the "excitation layer"
            self.activation_se1,
            layers.Dense(units=f3, use_bias=False),
            self.activation_se2,
            layers.Reshape([1, 1, f3]) #So we can multiply to the input
        ])

        channel_increase = not (input_shape[-1] == f3)
        if channel_increase:
            self.id_pass = tf.keras.models.Sequential([
                layers.Conv2D(f3, 1, strides=self.stride),
                layers.BatchNormalization(),
            ])
        else:
            self.id_pass = layers.BatchNormalization()

        self.final_pass = self.activation3

    def call(self, input):
        y = self.normal_pass(input)
        #print(y.shape)
        yy = self.excitation(y)#bc this works better sighhhhh
        #print(yy.shape)

        return self.final_pass((y * yy) + self.id_pass(input))

print("\nConstructing Model")
strat = tf.distribute.MirroredStrategy()
print("\t{} Available Devices".format(strat.num_replicas_in_sync))

with strat.scope():
    optim = tf.keras.optimizers.experimental.SGD(INIT_LRATE, MOMENTUM, weight_decay=(W_DECAY if W_DECAY > 0 else None))
    #optim = tf.keras.optimizers.AdamW(INIT_LRATE, W_DECAY, use_ema=(False if MOMENTUM <= 0 else True), ema_momentum=MOMENTUM)
    replacement = layers.Activation if MODEL_T == "control" else (II.PiecewiseLinearUnitV1 if MODEL_T == "PWLU" else II.ActivationLinearizer)
    
    args_relu = ["relu"] if MODEL_T != "PWLU" else []
    args_sigmoid = ["sigmoid"] if MODEL_T != "PWLU" else []

    def produce_SE(filters, stride=1):
        return SE_ResidualBlock(
            filters, 
            stride, 
            a1=replacement(*args_relu),
            a2=replacement(*args_relu),
            a3=replacement(*args_relu),
            ase1=replacement(*args_relu),
            ase2=replacement(*args_sigmoid)
        )

    se_net = models.Sequential([
        layers.Conv2D(64, 7, strides=2, input_shape=(*IMAGE_SIZE, 3)),
        layers.BatchNormalization(),
        replacement(*args_relu),
        layers.MaxPooling2D(3, 2),

        produce_SE((64, 64, 256)),
        produce_SE((64, 64, 256)),
        produce_SE((64, 64, 256)),

        produce_SE((128, 128, 512), stride=2),
        produce_SE((128, 128, 512)),
        produce_SE((128, 128, 512)),

        produce_SE((256, 256, 1024), stride=2),
        produce_SE((256, 256, 1024)),
        produce_SE((256, 256, 1024)),
        produce_SE((256, 256, 1024)),
        produce_SE((256, 256, 1024)),
        produce_SE((256, 256, 1024)),

        produce_SE((512, 512, 2048), stride=2),
        produce_SE((512, 512, 2048)),
        produce_SE((512, 512, 2048)),

        layers.AveragePooling2D(2, padding='same'),
        layers.Flatten(),

        layers.Dense(256),
        replacement(*args_relu),

        layers.Dense(128),
        replacement(*args_relu),

        layers.Dense(1000 if DATASET == "imagenet2012" else 10, activation='softmax')
	])

    metrics_to_use = []
    if DATASET == "imagenet2012":
        metrics_to_use.extend([
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=500, name="T500"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=250, name="T250")
        ])
    
    metrics_to_use.extend([
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="T5"),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="T3"),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1, name="T1")
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

#se_net.summary(line_length=80)

print("\nStarting training")
se_net.compile(optim, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=metrics_to_use)
hist = se_net.fit(train_ds, epochs=MAX_EPOCH, validation_data=val_ds, callbacks=calls)

print("\n")
logger.info("{} {} Training History".format(DATASET, MODEL_T))
if DATASET != 'cifar-10' and DATASET != 'imagenette':
	logger.info("Training T500: {}".format(hist.history["T500"]))
	logger.info("Training T250: {}".format(hist.history["T250"]))
logger.info("Training T5: {}".format(hist.history["T5"]))
logger.info("Training T3: {}".format(hist.history["T3"]))
logger.info("Training T1: {}".format(hist.history["T1"]))
logger.info("Training Loss: {}".format(hist.history["loss"]))

print("\n")
logger.info("\n")
if DATASET != 'cifar-10' and DATASET != 'imagenette':
	logger.info("Validate T500: {}".format(hist.history["val_T500"]))
	logger.info("Validate T250: {}".format(hist.history["val_T250"]))
logger.info("Validate T5: {}".format(hist.history["val_T5"]))
logger.info("Validate T3: {}".format(hist.history["val_T3"]))
logger.info("Validate T1: {}".format(hist.history["val_T1"]))
logger.info("Validate Loss: {}".format(hist.history["val_loss"]))