import sys
import logging
import tensorflow as tf
import tensorflow_datasets as tfds
import time
import InferredActivations.Inferrer as II
from keras import layers
from keras.metrics import TopKCategoricalAccuracy

#Logging Set up
logger = logging.getLogger("internal_Logger_wo__imports")
logger.setLevel(logging.DEBUG)
fileHandle = logging.FileHandler(filename="alexnet_testing_suite.log")
streamHandle = logging.StreamHandler(sys.stdout)

formatter = logging.Formatter(fmt='%(message)s')
fileHandle.setFormatter(formatter)
streamHandle.setFormatter(formatter)

logger.addHandler(fileHandle)
#logger.addHandler(streamHandle)

TOTAL_EPOCHS = 10
BATCH_SIZE = 32
IMAGE_SIZE = (227, 227, 3)
DROPOUT_RATE = 0.25
AUGMENT_DATA = True
CONCATENATE_AUGMENT = True

print("Starting AlexNet Sandbox")
print("\t{} Epochs\n\tBatched in {}\n\tDropout Rate of {}".format(TOTAL_EPOCHS, BATCH_SIZE, DROPOUT_RATE))
print("\nLoading imagenette")
train_ds, val_ds = tfds.load("imagenette", split=['train', 'validation'], as_supervised=True, batch_size=BATCH_SIZE)

resize_and_rescale = tf.keras.models.Sequential([
    layers.Resizing(IMAGE_SIZE[0], IMAGE_SIZE[1]),
    layers.Rescaling(1./255),
])

def preprocess1(x, y):
    return resize_and_rescale(x, training=True), y

train_data = train_ds.map(preprocess1)
val_data = val_ds.map(preprocess1)

if AUGMENT_DATA:
    print("\tAugmenting Data")
    data_augmentation = tf.keras.models.Sequential([
        #layers.RandomZoom((-0.5, -0.25), (-0.5, -0.25)),
        layers.RandomZoom((0, -0.25)),
        layers.Resizing(IMAGE_SIZE[0]*2, IMAGE_SIZE[1]*2),
        layers.RandomCrop(IMAGE_SIZE[0], IMAGE_SIZE[0]),
        layers.RandomFlip(),
        resize_and_rescale
    ])

    def preprocess2(x, y):
        return data_augmentation(x, training=True), y
    
    if CONCATENATE_AUGMENT:
        print("\tConcatenating Augment")
        train_data = train_data.concatenate(train_ds.map(preprocess2))
    else:
        print("\tReplacing Original with Augment")
        train_data = train_ds.map(preprocess2)

#Define the Model
print("\nDefining The Model")
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

replaceable_activations_1 = [1, 4, 7, 9, 11, 15, 18] #Normal relu type beat
replaceable_activations_2 = [21] #Softmax type beat

print("\nEstablishing Multi-GPU Target")
strat = tf.distribute.MirroredStrategy()
print("\tAvailable Devices: {}".format(strat.num_replicas_in_sync))

print("\nBuilding AlexNet Models")
with strat.scope():
    metrics_to_use = [TopKCategoricalAccuracy(name="T5"), TopKCategoricalAccuracy(k=3, name="T3"), TopKCategoricalAccuracy(k=1, name="T1")]
    cllbcks_to_use = [tf.keras.callbacks.TerminateOnNaN()]

    buildables = [
        {
            "name": "Control-Alex",
            "replacement": layers.Activation,
            "args": ['relu'],
            "indices_of": []
        },
        {
            "name": "FullPLU-Alex",
            "replacement": II.PiecewiseLinearUnitV1,
            "args": [],
            "indices_of": [1, 4, 7, 9, 11, 15, 18]
        },
        {
            "name": "HalfBotPLU-Alex",
            "replacement": II.PiecewiseLinearUnitV1,
            "args": [],
            "indices_of": [1, 4, 7, 9]
        },
        {
            "name": "HalfTopPLU-Alex",
            "replacement": II.PiecewiseLinearUnitV1,
            "args": [],
            "indices_of": [9, 11, 15, 18]
        },
        {
            "name": "FullAL-Alex",
            "replacement": II.ActivationLinearizer,
            "args": ['relu'],
            "indices_of": [1, 4, 7, 9, 11, 15, 18]
        },
        {
            "name": "HalfBotAL-Alex",
            "replacement": II.ActivationLinearizer,
            "args": ['relu'],
            "indices_of": [1, 4, 7, 9]
        },
        {
            "name": "HalfTopAL-Alex",
            "replacement": II.ActivationLinearizer,
            "args": ['relu'],
            "indices_of": [9, 11, 15, 18],
        },
    ]

    models_to_test = []
    for k in buildables:
        new_model = tf.keras.models.Sequential(name=k["name"])

        for i in range(len(basic_alex_net_definition)):
            if not (i in k["indices_of"]):
                lyrFn = basic_alex_net_definition[i]["lyrFn"]
                args = basic_alex_net_definition[i]["args"]
                kwargs = basic_alex_net_definition[i]["kwargs"]

                new_model.add(lyrFn(*args, **kwargs))
            else:
                new_model.add(k["replacement"](*k["args"]))
        
        new_model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=metrics_to_use)
        new_model.build((None, *IMAGE_SIZE))
        models_to_test.append(new_model)

print("\nBeginning Testing Suite")
for model in models_to_test:
    print("\tTraining " + model.name)
    fit_then = time.time()
    fit_data = model.fit(train_data, epochs=TOTAL_EPOCHS, verbose=0, validation_data=val_data, callbacks=cllbcks_to_use)
    fit_time = time.time() - fit_then

    print("\n\t\tEvaluating " + model.name)

    eval_then = time.time()
    eval_data = model.evaluate(val_data, verbose=0)
    eval_time = time.time() - eval_then

    #Best And Worst Epochs
    BestEpoch, BestTop1 = 0, 0
    BestTop3, BestTop5 = 0, 0

    WorsEpoch, WorsTop1 = 0, 1
    WorsTop3, WorsTop5 = 1, 1

    for i in range(TOTAL_EPOCHS):
        if BestTop1 < fit_data.history["val_T1"][i]:
            BestEpoch = i
            BestTop1 = fit_data.history["val_T1"][i]
        elif BestTop1 == 0 and BestEpoch == 0:
            if BestTop3 < fit_data.history["val_T3"][i]:
                BestEpoch = i
                BestTop3 = fit_data.history["val_T3"][i]
            elif BestTop3 == 0 and (BestTop5 < fit_data.history["val_T5"][i]):
                BestEpoch = i
                BestTop5 = fit_data.history["val_T5"][i]

        if WorsTop1 > fit_data.history["val_T1"][i]:
            WorsEpoch = i
            WorsTop1 = fit_data.history["val_T1"][i]

    final_log_info = model.name + " results:\n\tTest-Evaluation Top 5: " + str(eval_data[1]) + ", Top 3: " + str(eval_data[2]) + ", Top 1: "  + str(eval_data[-1])
    final_log_info += "\n\tFinal Training Accuracies were Top 5: " + str(fit_data.history["val_T5"][-1]) + ", Top 3: " + str(fit_data.history["val_T3"][-1]) + ", Top 1: " + str(fit_data.history["val_T1"][-1])
    final_log_info += "\n\tBest Training Epoch was (" + str(BestEpoch) + ") Top 5: " + str(fit_data.history["val_T5"][BestEpoch]) + ", Top 3: " + str(fit_data.history["val_T3"][BestEpoch]) + ", Top 1: "  + str(BestTop1)
    final_log_info += "\n\tWorst Training Epoch was (" + str(WorsEpoch) + ") Top 5: " + str(fit_data.history["val_T5"][WorsEpoch]) + ", Top 3: " + str(fit_data.history["val_T3"][WorsEpoch]) + ", Top 1: "  + str(WorsTop1)
    final_log_info += "\n\tTraining Time was: " + str(int(fit_time)) + "s, Evaluation Time was: " + str(int(eval_time)) + "s"
    final_log_info += "\n\tNaN Check: " + str(fit_data.history["loss"][-1]) + ", " + str(fit_data.history["val_loss"][-1])
    logger.info(final_log_info + "\n")