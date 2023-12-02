#NOTE: Copied a lot from SE_NetBigTest.py
import tensorflow as tf
import tensorflow_datasets as tfds
import logging
import InferredActivations.Inferrer as II

from keras import layers, models

#Setting up Logger
logger = logging.getLogger("internal_logger")
logger.setLevel(logging.DEBUG)
fileHandle = logging.FileHandler(filename="drop_in.log")

formatter = logging.Formatter(fmt='%(message)s')
fileHandle.setFormatter(formatter)
logger.addHandler(fileHandle)

def print_history(histo):
    logger.info("\nTraining T5: {}".format(histo.history["T5"]))
    logger.info("Training T3: {}".format(histo.history["T3"]))
    logger.info("Training T1: {}".format(histo.history["T1"]))
    logger.info("Training Loss: {}".format(histo.history["loss"]))

    logger.info("\nValidate T5: {}".format(histo.history["val_T5"]))
    logger.info("Validate T3: {}".format(histo.history["val_T3"]))
    logger.info("Validate T1: {}".format(histo.history["val_T1"]))
    logger.info("Validate Loss: {}".format(histo.history["val_loss"]))

print("AL Drop-in test beginning!")
#Dataset Information
DATASET = "cifar10"
AUGMENT = True
CONCAT_A = True
FACTOR_A = 0.1
#Training/Definition Information
REDUCTION_RATIO = 16
BATCH_SZ = 32 #cifar-10 128, imagenette 8
TEACHER_EPOCHS = 20
STUDENT_EPOCHS = 20
IMAGE_SIZE = (224, 224) #Assumed Channels last
#Using Learning Rate Scheduler
INIT_LRATE = 0.01 #0.6 initial by paper
LRATE_SCHED = 30 #30 epochs by paper
LRATE_RATIO = 0.1 #0.1 ratio by paper
#Using SGD Experimental Optimizer
MOMENTUM = 0.9 #0.9 by paper
W_DECAY = 0 #0 by paper

assert DATASET in ["imagenette", "cifar10"] #TODO: imagenet2012?
print(f"\t{DATASET} training set\n\tAugment? {AUGMENT}\n\tConcatenate Augment? {CONCAT_A}\n\t{FACTOR_A} Augment Factor")
print(f"\t{BATCH_SZ} Batches\n\t{TEACHER_EPOCHS} Teacher Epochs\n\t{STUDENT_EPOCHS} Student Epochs\n\t{REDUCTION_RATIO} Reduction Ratio")
print(f"\t{INIT_LRATE} Initial Learning Rate\n\t{LRATE_RATIO} Factor Every {LRATE_SCHED} Epochs")
print(f"\t{MOMENTUM} SGD Momentum\n\t{W_DECAY} Weight Decay")

print(f"\nPrepping {DATASET}")
train_ds, val_ds = None, None
if DATASET == "imagenette":
	train_ds, val_ds = tfds.load("imagenette", split=["train", "validation"], as_supervised=True, batch_size=BATCH_SZ)
elif DATASET == "cifar10":
	training, testing = tf.keras.datasets.cifar10.load_data()
	train_ds = tf.data.Dataset.from_tensor_slices(training).batch(BATCH_SZ)
	val_ds = tf.data.Dataset.from_tensor_slices(testing).batch(BATCH_SZ)

print("\t{} Images Loaded for Training".format(train_ds.cardinality() * BATCH_SZ))
print("\t{} Images Loaded for Validation/Testing".format(val_ds.cardinality() * BATCH_SZ))

#NOTE: With how many times I copy the same code, I should just set up a "utils" file
resize_and_rescale = models.Sequential([
	layers.Resizing(*IMAGE_SIZE),
	layers.Rescaling(1./255),
])

data_augmentation = models.Sequential([
	layers.RandomFlip(),
	layers.RandomContrast(FACTOR_A),
	layers.RandomBrightness(FACTOR_A),
	layers.RandomRotation(FACTOR_A),
	resize_and_rescale
])

def prepare_dataset(ds, augment=False):
	dsn = ds.map((lambda x, y: (resize_and_rescale(x, training=True), y)), num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

	if augment:
		augmenter = (lambda x, y: (data_augmentation(x, training=True), y))
		if CONCAT_A:
			dsn = dsn.concatenate(ds.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False))
		else:
			dsn = ds.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

	print("\tDataset Prepared")
	return dsn.prefetch(buffer_size=tf.data.AUTOTUNE)

train_ds = prepare_dataset(train_ds, augment=AUGMENT)
val_ds = prepare_dataset(val_ds)

print("\nDefining Special Layers")
@tf.keras.saving.register_keras_serializable("Custom_Layer")
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
                 ase2 = layers.Activation('sigmoid'),
                 **kwargs):
        super(SE_ResidualBlock, self).__init__(**kwargs)
        self.filters = filters
        self.stride = stride
        self.activation1 = a1
        self.activation2 = a2
        self.activation3 = a3
        self.activation_se1 = ase1
        self.activation_se2 = ase2
    
    def get_config(self):
        base_config = super().get_config()
        config = {
            "filters": self.filters,
            "stride": self.stride,
        }
        return {**base_config, **config}
        
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
        ], name = "normal_pass")

        self.excitation = models.Sequential([
            layers.GlobalAveragePooling2D(),

            layers.Dense(units=(f3/REDUCTION_RATIO), use_bias=False), #Rest of this is the "excitation layer"
            self.activation_se1,
            layers.Dense(units=f3, use_bias=False),
            self.activation_se2,
            layers.Reshape([1, 1, f3]) #So we can multiply to the input
        ], name = "excitation")

        channel_increase = not (input_shape[-1] == f3)
        if channel_increase:
            self.id_pass = tf.keras.models.Sequential([
                layers.Conv2D(f3, 1, strides=self.stride),
                layers.BatchNormalization(),
            ], name="id_pass_lyr")
        else:
            self.id_pass = layers.BatchNormalization()

        self.final_pass = self.activation3

    def call(self, input):
        y = self.normal_pass(input)
        #print(y.shape)
        yy = self.excitation(y)#bc this works better sighhhhh
        #print(yy.shape)

        return self.final_pass((y * yy) + self.id_pass(input))

print("\nDefining Teacher Model")
strat = tf.distribute.MirroredStrategy()
print("\t{} Available Devices".format(strat.num_replicas_in_sync))

with strat.scope():
    optim = tf.keras.optimizers.experimental.SGD(INIT_LRATE, MOMENTUM, weight_decay=(W_DECAY if W_DECAY > 0 else None))

    se_net = models.Sequential([
        layers.Conv2D(64, 7, strides=2, input_shape=(*IMAGE_SIZE, 3)),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(3, 2),

        SE_ResidualBlock((64, 64, 256)),
        SE_ResidualBlock((64, 64, 256)),
        SE_ResidualBlock((64, 64, 256)),

        SE_ResidualBlock((128, 128, 512), stride=2),
        SE_ResidualBlock((128, 128, 512)),
        SE_ResidualBlock((128, 128, 512)),

        SE_ResidualBlock((256, 256, 1024), stride=2),
        SE_ResidualBlock((256, 256, 1024)),
        SE_ResidualBlock((256, 256, 1024)),
        SE_ResidualBlock((256, 256, 1024)),
        SE_ResidualBlock((256, 256, 1024)),
        SE_ResidualBlock((256, 256, 1024)),

        SE_ResidualBlock((512, 512, 2048), stride=2),
        SE_ResidualBlock((512, 512, 2048)),
        SE_ResidualBlock((512, 512, 2048)),

        layers.AveragePooling2D(2, padding='same'),
        layers.Flatten(),

        layers.Dense(256),
        layers.Activation("relu"),

        layers.Dense(128),
        layers.Activation("relu"),

        layers.Dense(10, activation='softmax')
	])

    metrics_to_use = [
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="T5"),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="T3"),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1, name="T1")
    ]

    def learn_rate_scheduler(epoch, lr):
        interval_check = epoch % LRATE_SCHED
        if interval_check == 0:
            return lr * min(max(LRATE_RATIO, 0), 1)
        return lr
    
    calls = [
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.LearningRateScheduler(learn_rate_scheduler)
    ]

#se_net.summary(line_length=70)

print("\nAttempting to load control weights")
try:
    se_net.load_weights(f"./{DATASET}_se_net.h5")
    print("\tValidating weights")
    se_net.compile(optim, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=metrics_to_use)
    se_net.evaluate(val_ds)
except:
    print("\tDid not find control weights!")
    print("\tTraining Control")
    se_net.compile(optim, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=metrics_to_use)
    hist_t = se_net.fit(train_ds, epochs=TEACHER_EPOCHS, validation_data=val_ds, callbacks=calls)

    print_history(hist_t)

    se_net.save_weights(f"./{DATASET}_se_net.h5", save_format="h5")

print("\nDefining Student Model")
with strat.scope():
    #to reset optimizer state
    optim2 = tf.keras.optimizers.experimental.SGD(INIT_LRATE, MOMENTUM, weight_decay=(W_DECAY if W_DECAY > 0 else None))

    def produce_SE(filters, stride=1):
        return SE_ResidualBlock(
            filters,
            stride,
            II.ActivationLinearizer("relu"),
            II.ActivationLinearizer("relu"),
            II.ActivationLinearizer("relu"),
            II.ActivationLinearizer("relu"),
            II.ActivationLinearizer("sigmoid")
        )

    student_se_net = models.Sequential([
        layers.Conv2D(64, 7, strides=2, input_shape=(*IMAGE_SIZE, 3)),
        layers.BatchNormalization(),
        II.ActivationLinearizer("relu"),
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
        II.ActivationLinearizer("relu"),

        layers.Dense(128),
        II.ActivationLinearizer("relu"),

        layers.Dense(10, activation='softmax')
    ])

print("\nEvaluating Student Model")
student_se_net.load_weights(f"./{DATASET}_se_net.h5", skip_mismatch=True, by_name=True)
student_se_net.compile(optim2, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=metrics_to_use)
student_se_net.evaluate(val_ds)

print("\nTraining Student Model")
hist_s = student_se_net.fit(train_ds, epochs=STUDENT_EPOCHS, validation_data=val_ds, callbacks=calls)

print_history(hist_s)