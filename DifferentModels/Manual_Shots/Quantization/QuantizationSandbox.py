import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_model_optimization as tfmot
import InferredActivations.Inferrer as II
from QuantLib import DefaultCustomQuantizeConfig
from tensorflow_model_optimization.quantization.keras import quantize_annotate_layer, quantize_annotate_model
from keras.metrics import TopKCategoricalAccuracy
from keras import layers, models

TOTAL_EPOCHS = 50
BATCH_SIZE = 32
IMAGE_SIZE = (227, 227, 3)
DROPOUT_RATE = 0.5
N_BITS = 8 #Not currently being used
QUANTIZE = False

AUGMENT_DATA = True
CONCATENATE_AUGMENT = False

print("Starting Quantization Sandbox")
print("\t{} Total Epochs\n\t{} Batch Size\n\t{} Dropout\n\tQuantized? {}\n\t\tw/ {} Bits".format(TOTAL_EPOCHS, BATCH_SIZE, DROPOUT_RATE, QUANTIZE, N_BITS))

print("\nLoading in Imagenette/320px-v2")
train_base, val_base = tfds.load("imagenette/320px-v2", split=["train", "validation"], as_supervised=True, batch_size=BATCH_SIZE)

resize_and_rescale = tf.keras.models.Sequential([
    layers.Resizing(IMAGE_SIZE[0], IMAGE_SIZE[1]),
    layers.Rescaling(1./255),
])

data_augmentation = tf.keras.models.Sequential([
    #layers.RandomZoom((-0.05, -0.5)),
    layers.RandomContrast(factor=0.25),
    layers.RandomBrightness((-0.25, 0.25)),
    #layers.RandomFlip(),
    resize_and_rescale
])

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
            dsn = dsn.concatenate(ds.map(preprocess2))
        else:
            print("\t\tReplacing Original with Augment")
            dsn = ds.map(preprocess2)

    dsn.batch(BATCH_SIZE)

    print("\tDataset Prepared")
    return dsn.prefetch(buffer_size=tf.data.AUTOTUNE) #Sometimes, it's the simple things that make all the difference

train_data = prepare_dataset(train_base, AUGMENT_DATA)
val_data = prepare_dataset(val_base)

print("\nBuilding The Model")
print("\tEstablishing Multi-GPU Target")
strat = tf.distribute.MirroredStrategy()
print("\t\tAvailable Devices: {}".format(strat.num_replicas_in_sync))

#with strat.scope():
metrics_to_use = [TopKCategoricalAccuracy(name="T5"), TopKCategoricalAccuracy(k=3, name="T3"), TopKCategoricalAccuracy(k=1, name="T1")]

alex_net = models.Sequential([
    layers.Conv2D(96, 11, strides=(4, 4), input_shape=IMAGE_SIZE),
    layers.Activation('relu'),
    layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2)),

    layers.Conv2D(256, 5, padding='same'),
    layers.Activation('relu'),
    layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2)),

    layers.Conv2D(384, 3, padding='same'),
    layers.Activation('relu'),
    layers.Conv2D(384, 3, padding='same'),
    layers.Activation('relu'),
    layers.Conv2D(256, 3, padding='same'),
    layers.Activation('relu'),
    layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2)),
    layers.Flatten(),

    layers.Dense(4096),
    layers.Activation('relu'),

    layers.Dropout(DROPOUT_RATE),
    layers.Dense(4096),
    layers.Activation('relu'),

    layers.Dropout(DROPOUT_RATE),
    layers.Dense(1000),
    layers.Activation('relu'),

    layers.Dense(10),
    layers.Activation('softmax'),
], name = "alex_net")

if QUANTIZE:
    print("\tQuantizing Model")
    alex_net = tfmot.quantization.keras.quantize_model(alex_net)

alex_net.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=metrics_to_use)

print("\nTraining Model")
alex_net.fit(train_data, epochs=TOTAL_EPOCHS, validation_data=val_data)

#5 Epochs, Best Scores
#Control Normal: T5 100% T3 __0% T1 __0%
#Control Quante: T5 100% T3 100% T1 __0%

#PLU Full Norml: T5 _51% T3 __1% T1 __0% 
#PLU Full Quant: T5 100% T3 100% T1 100% (NOTE: Got T1 100% at E1, then Loss started growing massively to over 30k+)

#AL Full Normal: T5 100% T3 100% T1 __0% (4)
#AL Full Quante: T5 100% T3 100% T1 __0%


#5 Epochs, Last Scores, Best Scores
#Control Normal: T5 100% T3 100% T1 __0% Final | T5 100% T3 100% T1 __0% Epoch-7
#Control Quante: T5 100% T3 100% T1 __0% Final | T5 100% T3 100% T1 __0% Epoch-1

#PLU Full Norml: T5 _41% T3 _27% T1 _12% Final | T5 _68% T3 _61% T1 _45% Epoch-18
#PLU Full Quant: T5 100% T3 100% T1 100% Final | T5 100% T3 100% T1 100% Epoch-8  (NOTE: Loss can reach a max of 436,763,099,136 and is inconsistent)

#AL Full Normal: T5 100% T3 100% T1 __0% Final | T5 100% T3 100% T1 __0% Epoch-2
#AL Full Quante: T5 100% T3 100% T1 100% Final | T5 100% T3 100% T1 100% Epoch-4