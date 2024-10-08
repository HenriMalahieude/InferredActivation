import tensorflow as tf
import tensorflow_datasets as tfds
import logging
import InferredActivations.Inferrer as II
from keras import layers, models

#Logging Set up
logger = logging.getLogger("internal_Logger_wo__imports")
logger.setLevel(logging.DEBUG)
fileHandle = logging.FileHandler(filename="alexnet_imagenet_dump.log")

formatter = logging.Formatter(fmt='%(message)s')
fileHandle.setFormatter(formatter)

logger.addHandler(fileHandle)

#Quiets 
#"2023-07-28 18:38:33.330313: I tensorflow/compiler/xla/stream_executor/cuda/cuda_gpu_executor.cc:996] successful NUMA node read from SysFS had negative value (-1), 
# but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355"
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

MAX_EPOCH = 15
BATCH_SIZE = 8
INIT_LRATE = 0.01 #AlexNet Paper used 0.01 initially
LRATE_EPOCH_DEC = 5
LRATE_DECREASE = 0.1 # divide by 10 from AlexNet Paper
MOMENTUM = 0.9 #0.9 From AlexNet Paper
WEIGHT_DECAY = 0.0005 #0.0005 From AlexNet Paper

IMAGE_SIZE = (227, 227, 3)
DROPOUT_RATE = 0.5

AUGMENT_DATA = True
CONCATENATE_AUGMENT = False
AUGMENT_FACTOR = 0.1

DATASET = "imagenet2012"
assert DATASET in ["imagenet2012", "CIFAR-10", "Imagenette"]

ACTS = "control"
assert ACTS in ["control", "PLU", "AL"]

print("Starting AlexNet Big Test, Stats:")
print("\t{} Batch Size\n\t({}, {}, {}) Image Size\n\t{} Dropout Rate".format(BATCH_SIZE, IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE[2], DROPOUT_RATE))
print("\t{} Max Epochs\n\t{} Augment Factor".format(MAX_EPOCH, AUGMENT_FACTOR))
print("\t{} dataset\n\t{} Activations".format(DATASET, ACTS))

print("\nPrepping Dataset")
train_ds, val_ds = None, None
if DATASET == "imagenet2012":
    print("\tLoading in imagenet2012 dataset")
    train_ds, val_ds = tfds.load("imagenet2012", split=["train", "validation"], as_supervised=True, batch_size=BATCH_SIZE)
elif DATASET == "CIFAR-10":
    print("\tLoading in CIFAR-10 dataset")
    training, testing = tf.keras.datasets.cifar10.load_data()
    train_ds = tf.data.Dataset.from_tensor_slices(training).batch(BATCH_SIZE)
    val_ds = tf.data.Dataset.from_tensor_slices(testing).batch(BATCH_SIZE)
elif DATASET == "Imagenette":
    print("\tLoading in Imagenette dataset")
    train_ds, val_ds = tfds.load("imagenette", split=["train", "validation"], as_supervised=True, batch_size=BATCH_SIZE)

print("\t" + str(train_ds.cardinality() * BATCH_SIZE) + " training images loaded.")
print("\t" + str(val_ds.cardinality() * BATCH_SIZE) + " validation images loaded.")
#print("\t" + str(test_ds.cardinality() * BATCH_SIZE) + " testing images loaded.")

"""
Olga Russakovsky*, Jia Deng*, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, 
Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg and Li Fei-Fei. (* = equal contribution) 
ImageNet Large Scale Visual Recognition Challenge. arXiv:1409.0575, 2014.
"""

resize_and_rescale = models.Sequential([
    layers.Resizing(IMAGE_SIZE[0], IMAGE_SIZE[1]),
    layers.Rescaling(1./255),
])

data_augmentation = models.Sequential([
    layers.Resizing(256, 256),
    layers.RandomCrop(IMAGE_SIZE[0], IMAGE_SIZE[1]),
    #layers.RandomZoom((-0.1, -0.5)),
    layers.RandomFlip("horizontal"),
    layers.RandomContrast(AUGMENT_FACTOR),
    layers.RandomBrightness((-1 * AUGMENT_FACTOR, AUGMENT_FACTOR)),
    resize_and_rescale
])

print("\tPreprocessing Datasets")
def preprocess1(x, y):
    return resize_and_rescale(x, training=True), y

def preprocess2(x, y):
    return data_augmentation(x, training=True), y

def prepare_dataset(ds, augment=False):
    dsn = ds.map(preprocess1, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    
    if augment:
        print("\tAugmenting Data")
        if CONCATENATE_AUGMENT:
            print("\t\tConcatenating Augment")
            dsn = dsn.concatenate(ds.map(preprocess2, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False))
        else:
            print("\t\tReplacing Original with Augment")
            dsn = ds.map(preprocess2, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

    print("\tDataset Prepared")
    return dsn.prefetch(buffer_size=tf.data.AUTOTUNE)

train_data = prepare_dataset(train_ds, AUGMENT_DATA)
val_data = prepare_dataset(val_ds)

print("\nCreating Mirrored Multi-GPU Strategy")
strat = tf.distribute.MirroredStrategy()
print("\tNumber of Devices: {}".format(strat.num_replicas_in_sync))

with strat.scope():
    optim = tf.keras.optimizers.AdamW(INIT_LRATE, WEIGHT_DECAY, use_ema=True, ema_momentum=MOMENTUM)

    inits = {
        "zero-gaussian" : tf.keras.initializers.RandomNormal(0.0, 0.01, seed=15023),
        "const1": tf.keras.initializers.Constant(1),
        "const0": tf.keras.initializers.Constant(0),
    }
    
    kwargs = [
        {"kernel_initializer": inits["zero-gaussian"], "bias_initializer": inits["const0"]},
        {"kernel_initializer": inits["zero-gaussian"], "bias_initializer": inits["const1"]},
    ]

    replacement = II.ActivationLinearizer if ACTS == "AL" else II.PiecewiseLinearUnitV1

    alex_net = tf.keras.models.Sequential([
        layers.Conv2D(96, 11, strides=(4, 4), input_shape=IMAGE_SIZE, **kwargs[0]),
        layers.Activation('relu') if ACTS == "control" else replacement(),
        layers.MaxPooling2D(pool_size=(3, 3), strides=(2,2)),

        layers.Conv2D(256, 5, padding='same', **kwargs[1]),
        layers.Activation('relu') if ACTS == "control" else replacement(),
        layers.MaxPooling2D(pool_size=(3,3), strides=(2,2)),

        layers.Conv2D(384, 3, padding='same', **kwargs[1]),
        layers.Activation('relu') if ACTS == "control" else replacement(),
        layers.Conv2D(384, 3, padding='same', **kwargs[1]),
        layers.Activation('relu') if ACTS == "control" else replacement(),
        layers.Conv2D(256, 3, padding='same', **kwargs[0]),
        layers.Activation('relu') if ACTS == "control" else replacement(),
        layers.MaxPooling2D(pool_size=(3, 3), strides=(2,2)),

        layers.Flatten(),
        layers.Dense(4096, **kwargs[0]),
        layers.Activation('relu') if ACTS == "control" else replacement(),
        layers.Dropout(DROPOUT_RATE),

        layers.Dense(4096, **kwargs[0]),
        layers.Activation('relu') if ACTS == "control" else replacement(),
        layers.Dropout(DROPOUT_RATE),

        layers.Dense(1000 if DATASET == "imagenet2012" else 10, **kwargs[0]),
        layers.Activation('softmax')
    ])

    def learn_rate_scheduler(epoch, lr):
        intervals = MAX_EPOCH // LRATE_EPOCH_DEC
        
        for i in range(intervals):
            if ((i+1) * LRATE_EPOCH_DEC) == epoch:
                return lr * min(max(LRATE_DECREASE, 0), 1)
        
        return lr

    checks_to_use = [
        #tf.keras.callbacks.EarlyStopping(monitor='val_T5', patience=20, mode='max', start_from_epoch=5, verbose=1),
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.LearningRateScheduler(learn_rate_scheduler, verbose=1)
    ]

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

print("\nBeginning Training")
alex_net.compile(optimizer=optim, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=metrics_to_use)
hist = alex_net.fit(train_data, epochs=MAX_EPOCH, validation_data=val_data, callbacks=checks_to_use)

logger.info("{} {} History: ".format(DATASET, ACTS))
logger.info("Training Loss: {}".format(hist.history["loss"]))
if DATASET == "imagenet2012":
    logger.info("Training T500: {}".format(hist.history["T500"]))
    logger.info("Training T250: {}".format(hist.history["T250"]))
logger.info("Training T5: {}".format(hist.history["T5"]))
logger.info("Training T3: {}".format(hist.history["T3"]))
logger.info("Training T1: {}".format(hist.history["T1"]))

logger.info("\nValidation Loss: {}".format(hist.history["val_loss"]))
if DATASET == "imagenet2012":
    logger.info("Validation T500: {}".format(hist.history["val_T500"]))
    logger.info("Validation T250: {}".format(hist.history["val_T250"]))
logger.info("Validation T5: {}".format(hist.history["val_T5"]))
logger.info("Validation T3: {}".format(hist.history["val_T3"]))
logger.info("Validation T1: {}".format(hist.history["val_T1"]))