import time
import logging
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.quantization.keras import quantize_annotate_layer, quantize_annotate_model
import InferredActivations.Inferrer as II
from keras import layers, models
from keras.metrics import TopKCategoricalAccuracy


#Logging Set up
logger = logging.getLogger("internal_Logger_wo__imports")
logger.setLevel(logging.DEBUG)
fileHandle = logging.FileHandler(filename="alexquantized_testing_suite.log")

formatter = logging.Formatter(fmt='%(message)s')
fileHandle.setFormatter(formatter)

logger.addHandler(fileHandle)

TOTAL_EPOCHS = 1
BATCH_SIZE = 32
IMAGE_SIZE = (227, 227, 3)
AUGMENT_DATA = True
CONCATENATE_AUGMENT = False
DROPOUT_RATE = 0.5

print("\nInitiaing Quantization Suite for AlexNet")
print("\t\t{} Total Epochs\n\t\t{} Batch Size\n\t\t{} x {} Image Size\n\t\t{} Dropout".format(TOTAL_EPOCHS, BATCH_SIZE, IMAGE_SIZE[0], IMAGE_SIZE[1], DROPOUT_RATE))

print("\nLoading in Imagenette Dataset")
train_base, val_base = tfds.load("imagenette", split=["train", "validation"], as_supervised=True, batch_size=BATCH_SIZE)

resize_and_rescale = tf.keras.models.Sequential([
    layers.Resizing(IMAGE_SIZE[0], IMAGE_SIZE[1]),
    layers.Rescaling(1./255),
])

data_augmentation = tf.keras.models.Sequential([
    layers.RandomZoom((-0.05, -0.5)),
    layers.RandomContrast(factor=0.05),
    layers.RandomBrightness((-0.05, 0.05)),
    layers.RandomFlip(),
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

print("\nModel Definition")
#May need to define specific layers
basic_alex_net_definition = [
    #0
    {
        "lyrFn": layers.Conv2D,
        "args": [96, 11],
        "kwargs": {"strides": (4, 4)}
    },

    #1
    {
        "lyrFn": layers.Activation,
        "args": ['relu'],
        "kwargs": {}
    },

    #2
    {
        "lyrFn": layers.AveragePooling2D,
        "args": [],
        "kwargs": {"pool_size": (3, 3), "strides": (2,2)}
    },

    #3
    {
        "lyrFn": layers.Conv2D,
        "args": [256, 5],
        "kwargs": {"padding":'same'}
    },

    #4
    {
        "lyrFn": layers.Activation,
        "args": ['relu'],
        "kwargs": {}
    },

    #5
    {
        "lyrFn": layers.AveragePooling2D,
        "args": [],
        "kwargs": {"pool_size": (3, 3), "strides": (2,2)}
    },

    #6
    {
        "lyrFn": layers.Conv2D,
        "args": [384, 3],
        "kwargs": {"padding":'same'}
    },

    #7
    {
        "lyrFn": layers.Activation,
        "args": ['relu'],
        "kwargs": {}
    },

    #8
    {
        "lyrFn": layers.Conv2D,
        "args": [384, 3],
        "kwargs": {"padding":'same'}
    },

    #9
    {
        "lyrFn": layers.Activation,
        "args": ['relu'],
        "kwargs": {}
    },

    #10
    {
        "lyrFn": layers.Conv2D,
        "args": [384, 3],
        "kwargs": {"padding":'same'}
    },

    #11
    {
        "lyrFn": layers.Activation,
        "args": ['relu'],
        "kwargs": {}
    },

    #12
    {
        "lyrFn": layers.AveragePooling2D,
        "args": [],
        "kwargs": {"pool_size": (3, 3), "strides": (2,2)}
    },

    #13
    {
        "lyrFn": layers.Flatten,
        "args": [],
        "kwargs": {}
    },

    #14
    {
        "lyrFn": layers.Dense,
        "args": [4096],
        "kwargs": {}
    },

    #15
    {
        "lyrFn": layers.Activation,
        "args": ['relu'],
        "kwargs": {}
    },
    
    #16
    {
        "lyrFn": layers.Dropout,
        "args": [DROPOUT_RATE],
        "kwargs": {}
    },

    #17
    {
        "lyrFn": layers.Dense,
        "args": [4096],
        "kwargs": {}
    },

    #18
    {
        "lyrFn": layers.Activation,
        "args": ["relu"],
        "kwargs": {}
    },

    #19
    {
        "lyrFn": layers.Dropout,
        "args": [DROPOUT_RATE],
        "kwargs": {}
    },

    #20
    {
        "lyrFn": layers.Dense,
        "args": [10],
        "kwargs": {}
    },

    #21
    {
        "lyrFn": layers.Activation,
        "args": ["softmax"],
        "kwargs": {}
    },
]

activation_indices_ro = [1, 4, 7, 9, 11, 15, 18] #Other than softmax

buildables = [
    {
        "name": "Control",
        "quantized": False,
        "lyr_indices": [],
        #"lyr_replacement": layers.Activation,
        #"lyr_args": ["relu"],
        #"lyr_kwargs": {},
    },

    {
        "name": "Control-Q",
        "quantized": True,
        "lyr_indices": [],
        #"lyr_replacement": layers.Activation,
        #"lyr_args": ["relu"],
        #"lyr_kwargs": {},
    },

    {
        "name": "FullPLU",
        "quantized": False,
        "lyr_indices": [1, 4, 7, 9, 11, 15, 18],
        "lyr_replacement": II.PiecewiseLinearUnitV1,
        "lyr_args": [],
        "lyr_kwargs": {},
    },

    {
        "name": "FullPLU-Q",
        "quantized": True,
        "lyr_indices": [1, 4, 7, 9, 11, 15, 18],
        "lyr_replacement": II.PiecewiseLinearUnitV1,
        "lyr_args": [],
        "lyr_kwargs": {},
    },

    {
        "name": "FullAL",
        "quantized": False,
        "lyr_indices": [1, 4, 7, 9, 11, 15, 18],
        "lyr_replacement": II.ActivationLinearizer,
        "lyr_args": [],
        "lyr_kwargs": {},
    },

    {
        "name": "FullAL-Q",
        "quantized": True,
        "lyr_indices": [1, 4, 7, 9, 11, 15, 18],
        "lyr_replacement": II.ActivationLinearizer,
        "lyr_args": [],
        "lyr_kwargs": {},
    },
]

print("\nBuild the Models")
print("\tEstablish Multi-GPU Target")
strat = tf.distribute.MirroredStrategy()
print("\t\tAvailable Devices: {}".format(strat.num_replicas_in_sync))

print("\tBuilding:")
with strat.scope():
    metrics_to_use = [TopKCategoricalAccuracy(name="T5"), TopKCategoricalAccuracy(k=3, name="T3"), TopKCategoricalAccuracy(k=1, name="T1")]
    cllbcks_to_use = [tf.keras.callbacks.TerminateOnNaN()]

    models_to_test = []
    for k in buildables:
        print("\t\t" + k["name"])
        new_model = models.Sequential(name=k["name"])

        for i in range(len(basic_alex_net_definition)):
            lyrFn = basic_alex_net_definition[i]["lyrFn"]
            args = basic_alex_net_definition[i]["args"]
            kwargs = basic_alex_net_definition[i]["kwargs"]

            if i in k["lyr_indices"]: #Replace that layer
                lyrFn = k["lyr_replacement"]
                args = k["lyr_args"]
                kwargs = k["lyr_kwargs"]

            if k["quantized"]:
                with tfmot.quantization.keras.quantize_scope():
                    new_model.add(quantize_annotate_layer(lyrFn(*args, **kwargs)))
            else:
                new_model.add(lyrFn(*args, **kwargs))

        new_model.build((None, *IMAGE_SIZE))

        if k["quantized"]:
            new_model = tfmot.quantization.keras.quantize_model(new_model)
        
        new_model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=metrics_to_use)        
        models_to_test.append(new_model)

print("\nTraining:")
def formatPercentage(float):
    return int(float * 10000) / 100

start_at = 0 #for resuming when crashed, or annoyingly disconnected
for i in range(len(models_to_test)-start_at):
    model = models_to_test[i+start_at]
    print("\t" + model.name)
    fit_then = time.time()
    fit_data = model.fit(train_data, epochs=TOTAL_EPOCHS, verbose=1, validation_data=val_data, callbacks=cllbcks_to_use)
    fit_time = time.time() - fit_then

    print("\n\t\tEvaluating " + model.name)
    eval_then = time.time()
    eval_data = model.evaluate(val_data, verbose=2)
    eval_time = time.time() - eval_then

    #Best And Worst Epochs
    BestEpoch, BestTop1 = 0, 0
    BestTotal = 0

    WorsEpoch, WorsTop5 = 0, 1
    WorsTotal = 3

    for i in range(TOTAL_EPOCHS):
        t5, t3, t1 = fit_data.history["val_T5"][i], fit_data.history["val_T3"][i], fit_data.history["val_T1"][i]
        total = t5 + t3 + t1

        if (total > BestTotal or (BestTotal - total) < 0.05):
            if BestTop1 < 0.05 or t1 > BestTop1:
                BestEpoch = i
                BestTotal = total
                BestTop1 = t1

        if (total <= WorsTotal or (WorsTotal - total) < 0.05):
            if WorsTop5 > 0.95 or t5 <= WorsTop5:
                WorsEpoch = i
                WorsTotal = total
                WorsTop5 = t5

    final_log_info = model.name + " results:"
    #final_log_info += "\n\tTest-Evaluation Top 5: " + str(eval_data[1]) + ", Top 3: " + str(eval_data[2]) + ", Top 1: "  + str(eval_data[-1])
    final_log_info += "\n\tFVAs were T5: " + str(formatPercentage(fit_data.history["val_T5"][-1])) + "% T3: " + str(formatPercentage(fit_data.history["val_T3"][-1])) + "% T1: " + str(formatPercentage(fit_data.history["val_T1"][-1])) + "%"
    final_log_info += "\n\tBest Epoch was (" + str(BestEpoch+1) + ") w/ T5: " + str(formatPercentage(fit_data.history["val_T5"][BestEpoch])) + "% T3: " + str(formatPercentage(fit_data.history["val_T3"][BestEpoch])) + "% T1: "  + str(formatPercentage(fit_data.history["val_T1"][BestEpoch])) + "%"
    final_log_info += "\n\tWorst Epoch was (" + str(WorsEpoch+1) + ") w/ T5: " + str(formatPercentage(fit_data.history["val_T5"][WorsEpoch])) + "% T3: " + str(formatPercentage(fit_data.history["val_T3"][WorsEpoch])) + "% T1: "  + str(formatPercentage(fit_data.history["val_T1"][WorsEpoch])) + "%"
    final_log_info += "\n\tFinal Loss Train: " + str(fit_data.history["loss"][-1]) + ", Val: " + str(fit_data.history["val_loss"][-1])
    final_log_info += "\n\tFinal Training T5: " + str(formatPercentage(fit_data.history["T5"][-1])) + "% T3: " + str(formatPercentage(fit_data.history["T3"][-1])) + "% T1: " + str(formatPercentage(fit_data.history["T1"][-1])) + "%"
    final_log_info += "\n\tTraining Time was: " + str(int(fit_time)) + "s, Evaluation Time was: " + str(int(eval_time)) + "s"
    final_log_info += "\n"
    logger.info(final_log_info)


#Same thing