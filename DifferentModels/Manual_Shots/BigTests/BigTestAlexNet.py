import tensorflow as tf
import tensorflow_datasets as tfds
import time
import InferredActivations.Inferrer as II
from keras import layers

#Quiets 
#"2023-07-28 18:38:33.330313: I tensorflow/compiler/xla/stream_executor/cuda/cuda_gpu_executor.cc:996] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355"
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

BATCH_SIZE = 8
IMAGE_SIZE = (227, 227, 3)
DROPOUT_RATE = 0.6
AUGMENT_DATA = False
CONCATENATE_AUGMENTATION = False

print("Starting AlexNet Big Test, Stats:")
print("\t{} Batch Size\n\t({}, {}, {}) Image Size\n\t{} Dropout Rate".format(BATCH_SIZE, IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE[2], DROPOUT_RATE))
print("\nLoading in imagenet_sketch dataset")

#imagenet_ds_builder = tfds.builder(name="imagenet2012")
#imagenet_ds_builder.download_and_prepare(download_dir="~/tensorflow_datasets/downloads/manual")

train_ds, val_ds, test_ds = tfds.load("imagenet2012", split=["train", "validation", "test"], as_supervised=True, batch_size=BATCH_SIZE)

print("\t" + str(train_ds.cardinality() * BATCH_SIZE) + " training images loaded.")
print("\t" + str(val_ds.cardinality() * BATCH_SIZE) + " validation images loaded.")

"""
Olga Russakovsky*, Jia Deng*, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, 
Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg and Li Fei-Fei. (* = equal contribution) 
ImageNet Large Scale Visual Recognition Challenge. arXiv:1409.0575, 2014.
"""

resize_and_rescale = tf.keras.models.Sequential([
    layers.Resizing(IMAGE_SIZE[0], IMAGE_SIZE[1]),
    layers.Rescaling(1./255),
])

def preprocess1(x, y):
    return resize_and_rescale(x, training=True), y

train_data = train_ds.map(preprocess1)
val_data = val_ds.map(preprocess1)
test_data = test_ds.map(preprocess1)

if AUGMENT_DATA:
    print("\nAugmenting Data")
    augmentation = tf.keras.models.Sequential([
        layers.RandomZoom((0, -0.5)),
        layers.RandomFlip(),
        resize_and_rescale
    ])

    def preprocess2(x, y):
        return augmentation(x, training=True), y
    
    if CONCATENATE_AUGMENTATION:
        print("\tConcatenating Second Preprocess")
        train_data = train_data.concatenate(train_ds.map(preprocess2))
        print("\tIncreased Training Images to {}".format(train_data.cardinality() * BATCH_SIZE))
    else:
        train_data = train_ds.map(preprocess2)

for train_x, train_y in train_data:
    print("\tTrain Shape is TX:", train_x.shape, "TY:", train_y.shape)
    #print(train_y)
    break

print("\nCreating Mirrored Multi-GPU Strategy")
strat = tf.distribute.MirroredStrategy()
print("\tNumber of Devices: {}".format(strat.num_replicas_in_sync))

with strat.scope():
    alex_net = tf.keras.models.Sequential([
        layers.Conv2D(96, 11, strides=(4, 4), input_shape=IMAGE_SIZE),
        layers.Activation('relu'),
        layers.AveragePooling2D(pool_size=(3, 3), strides=(2,2)),

        layers.Conv2D(256, 5, padding='same'),
        layers.Activation('relu'),
        layers.AveragePooling2D(pool_size=(3,3), strides=(2,2)),

        layers.Conv2D(384, 3, padding='same'),
        layers.Activation('relu'),
        layers.Conv2D(384, 3, padding='same'),
        layers.Activation('relu'),
        layers.Conv2D(256, 3, padding='same'),
        layers.Activation('relu'),
        layers.AveragePooling2D(pool_size=(3, 3), strides=(2,2)),

        layers.Flatten(),
        layers.Dense(4096),
        layers.Activation('relu'),
        layers.Dropout(DROPOUT_RATE),

        layers.Dense(4096),
        layers.Activation('relu'),
        layers.Dropout(DROPOUT_RATE),

        layers.Dense(1000),
        layers.Activation('softmax')
    ])

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_T5', patience=20, mode='max', start_from_epoch=5, verbose=1)

    checks_to_use = [early_stop]
    metrics_to_use = [tf.keras.metrics.TopKCategoricalAccuracy(name="T5"), tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="T3"), tf.keras.metrics.TopKCategoricalAccuracy(k=1, name="T1")]

#alex_net.summary()

print("\nBeginning Training")
alex_net.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=metrics_to_use)
alex_net.fit(train_data, epochs=200, validation_data=val_data, callbacks=checks_to_use)

print("\n")
then = time.time()
alex_net.evaluate(test_data)
print("Took {}ms".format(int(((time.time() - then) * 1000))))

#-------------------------------100 Epochs with 25 Minimum Epochs
#Control ->