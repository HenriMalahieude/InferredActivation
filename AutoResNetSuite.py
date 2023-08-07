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
fileHandle = logging.FileHandler(filename="resnet_testing_suite.log")

formatter = logging.Formatter(fmt='%(message)s')
fileHandle.setFormatter(formatter)

logger.addHandler(fileHandle)

TOTAL_EPOCHS = 5
BATCH_SIZE = 64
IMAGE_SIZE = (227, 227, 3)
AUGMENT_DATA = True
CONCATENATE_AUGMENT = True

print("Starting Residual Auto Tester")
print("\t{} Epochs\n\tBatched in {}".format(TOTAL_EPOCHS, BATCH_SIZE))
print("\nLoading imagenette")
train_ds, val_ds = tfds.load("imagenette", split=['train', 'validation'], as_supervised=True, batch_size=BATCH_SIZE)

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

train_data = prepare_dataset(train_ds, AUGMENT_DATA)
val_data = prepare_dataset(val_ds)

print("\nModel Management")
print("\tDefining Residual Layer")
class ResidualBlock(layers.Layer):
    def __init__(self, 
                 filters = (64, 64, 256), 
                 activation_1 = layers.Activation("relu"),
                 activation_2 = layers.Activation("relu"),
                 activation_3 = layers.Activation("relu")):
        super(ResidualBlock, self).__init__()
        self.filters = filters
        self.a1 = activation_1
        self.a2 = activation_2
        self.a3 = activation_3

    def build(self, input_shape):
        f1, f2, f3 = self.filters
        channel_increase = not (input_shape[-1] == f3)
        stride = 1
        if channel_increase: stride = 2

        self.plain = tf.keras.models.Sequential([
            layers.Conv2D(f1, 1, strides=stride),
            layers.BatchNormalization(),
            self.a1,

            layers.Conv2D(f2, 3, padding='same'),
            layers.BatchNormalization(),
            self.a2,

            layers.Conv2D(f3, 1),
            layers.BatchNormalization(),
        ])

        
        if channel_increase:
            self.id_pass = tf.keras.models.Sequential([
                layers.Conv2D(f3, 1, strides=stride),
                layers.BatchNormalization(),
            ])
        else:
            self.id_pass = layers.BatchNormalization()

        self.final_pass = self.a3
    
    def call(self, input):
        return self.final_pass(self.plain(input) + self.id_pass(input))

print("\tDefining Control Model")
control_architecture = [
    #0
    {
        "lyrFn": layers.Conv2D,
        "args": [64, 7],
        "kwargs": {"strides": 2}
    },

    #1
    {
        "lyrFn": layers.BatchNormalization,
        "args": [],
        "kwargs": {},
    },

    #2
    {
        "lyrFn": layers.Activation,
        "args": ["relu"],
        "kwargs": {},
    },

    #3
    {
        "lyrFn": layers.MaxPooling2D,
        "args": [3, 2],
        "kwargs": {},
    },

    #4
    {
        "lyrFn": ResidualBlock,
        "args": [(64, 64, 256)],
        "kwargs": {},
    },

    #5
    {
        "lyrFn": ResidualBlock,
        "args": [(64, 64, 256)],
        "kwargs": {},
    },

    #6
    {
        "lyrFn": ResidualBlock,
        "args": [(64, 64, 256)],
        "kwargs": {},
    },

    #7
    {
        "lyrFn": ResidualBlock,
        "args": [(128, 128, 512)],
        "kwargs": {},
    },

    #8
    {
        "lyrFn": ResidualBlock,
        "args": [(128, 128, 512)],
        "kwargs": {},
    },

    #9
    {
        "lyrFn": ResidualBlock,
        "args": [(128, 128, 512)],
        "kwargs": {},
    },

    #10
    {
        "lyrFn": ResidualBlock,
        "args": [(256, 256, 1024)],
        "kwargs": {},
    },

    #11
    {
        "lyrFn": ResidualBlock,
        "args": [(256, 256, 1024)],
        "kwargs": {},
    },

    #12
    {
        "lyrFn": ResidualBlock,
        "args": [(256, 256, 1024)],
        "kwargs": {},
    },

    #13
    {
        "lyrFn": ResidualBlock,
        "args": [(256, 256, 1024)],
        "kwargs": {},
    },

    #14
    {
        "lyrFn": ResidualBlock,
        "args": [(256, 256, 1024)],
        "kwargs": {},
    },

    #15
    {
        "lyrFn": ResidualBlock,
        "args": [(256, 256, 1024)],
        "kwargs": {},
    },

    #16
    {
        "lyrFn": ResidualBlock,
        "args": [(512, 512, 2048)],
        "kwargs": {},
    },

    #17
    {
        "lyrFn": ResidualBlock,
        "args": [(512, 512, 2048)],
        "kwargs": {},
    },

    #18
    {
        "lyrFn": ResidualBlock,
        "args": [(512, 512, 2048)],
        "kwargs": {},
    },

    #19
    {
        "lyrFn": layers.AveragePooling2D,
        "args": [2],
        "kwargs": {"padding": "same"},
    },

    #20
    {
        "lyrFn": layers.Flatten,
        "args": [],
        "kwargs": {},
    },

    #21
    {
        "lyrFn": layers.Dense,
        "args": [256],
        "kwargs": {},
    },

    #22
    {
        "lyrFn": layers.Activation,
        "args": ["relu"],
        "kwargs": {},
    },

    #23
    {
        "lyrFn": layers.Dense,
        "args": [128],
        "kwargs": {},
    },

    #24
    {
        "lyrFn": layers.Activation,
        "args": ["relu"],
        "kwargs": {},
    },

    #25
    {
        "lyrFn": layers.Dense,
        "args": [10],
        "kwargs": {},
    },

    #26
    {
        "lyrFn": layers.Activation,
        "args": ["softmax"],
        "kwargs": {},
    },
]

#Mostly for easy access
raw_activation_indices = [2, 22, 24]
raw_softmax_index = [26]
residual_block_indices = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17] #4 - 18

print("\tDefining Buildables")
buildables = [
    {
        "name": "Control-Residual",
        "lyr_indices": [],
        "kwarg_indices": [],
    },

    {
        "name": "RawTopPLUs",
        "lyr_indices": [22, 24],
        "lyr_replacement": II.PiecewiseLinearUnitV1,
        "args_lyr": [],
        "kwargs_lyr": [],
        "kwarg_indices": [],
    },

    {
        "name": "PartialTopPLUs", #Inclusive
        "lyr_indices": [],
        "kwarg_indices": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        "kwarg_replacement": {"activation_2": II.PiecewiseLinearUnitV1(), "activation_3": II.PiecewiseLinearUnitV1()},
    },

    {
        "name": "PartialBotPLUs", #Inclusive
        "lyr_indices": [],
        "kwarg_indices": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        "kwarg_replacement": {"activation_1": II.PiecewiseLinearUnitV1(), "activation_2": II.PiecewiseLinearUnitV1()},
    },

    {
        "name": "FullResPLUs",
        "lyr_indices": [2, 22, 24],
        "lyr_replacement": II.PiecewiseLinearUnitV1,
        "args_lyr": [],
        "kwargs_lyr": [],
        "kwarg_indices": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        "kwarg_replacement": {"activation_1": II.PiecewiseLinearUnitV1(), "activation_2": II.PiecewiseLinearUnitV1(), "activation_3": II.PiecewiseLinearUnitV1()},
    },

    {
        "name": "RawTopALs",
        "lyr_indices": [22, 24],
        "lyr_replacement": II.ActivationLinearizer,
        "args_lyr": ["relu"],
        "kwargs_lyr": [],
        "kwarg_indices": [],
    },

    {
        "name": "PartialTopALs", #Inclusive
        "lyr_indices": [],
        "kwarg_indices": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        "kwarg_replacement": {"activation_2": II.ActivationLinearizer("relu"), "activation_3": II.ActivationLinearizer("relu")},
    },

    {
        "name": "PartialBotALs", #Inclusive
        "lyr_indices": [],
        "kwarg_indices": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        "kwarg_replacement": {"activation_1": II.ActivationLinearizer("relu"), "activation_2": II.ActivationLinearizer("relu")},
    },

    {
        "name": "FullResALs",
        "lyr_indices": [2, 22, 24],
        "lyr_replacement": II.ActivationLinearizer,
        "args_lyr": ["relu"],
        "kwargs_lyr": [],
        "kwarg_indices": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        "kwarg_replacement": {"activation_1": II.ActivationLinearizer("relu"), "activation_2": II.ActivationLinearizer("relu"), "activation_3": II.ActivationLinearizer("relu")},
    },
]

print("\tEstablishing Multi-GPU Target")
strat = tf.distribute.MirroredStrategy()
print("\t\tAvailable Devices: {}".format(strat.num_replicas_in_sync))

print("\tBuilding Models")
with strat.scope():
    metrics_to_use = [TopKCategoricalAccuracy(name="T5"), TopKCategoricalAccuracy(k=3, name="T3"), TopKCategoricalAccuracy(k=1, name="T1")]
    cllbcks_to_use = [tf.keras.callbacks.TerminateOnNaN()]
    
    models_to_test = []
    for k in buildables:
        new_model = tf.keras.models.Sequential(name=k["name"])

        for i in range(len(control_architecture)):
            lyrFn = control_architecture[i]["lyrFn"]
            args = control_architecture[i]["args"]
            kwargs = control_architecture[i]["kwargs"]

            if i in k["lyr_indices"]: #Replace that layer
                lyrFn = k["lyr_replacement"]
                args = k["args_lyr"]
                kwargs = k["kwargs_lyr"]
            elif i in k["kwarg_indices"]: #Replace their kwargs
                kwargs = k["kwarg_replacement"]
            
            new_model.add(lyrFn(*args, **kwargs))

        new_model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=metrics_to_use)
        new_model.build((None, *IMAGE_SIZE))
        models_to_test.append(new_model)

print("\nStarting Testing Suite")
start_at = 0 #for resuming when crashed, or annoyingly disconnected
for i in range(len(models_to_test)-start_at):
    model = models_to_test[i+start_at]
    print("\tTraining " + model.name)
    fit_then = time.time()
    fit_data = model.fit(train_data, epochs=TOTAL_EPOCHS, verbose=2, validation_data=val_data, callbacks=cllbcks_to_use)
    fit_time = time.time() - fit_then

    print("\n\t\tEvaluating " + model.name)
    eval_then = time.time()
    eval_data = model.evaluate(val_data, verbose=2)
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
        elif BestTop1 <= 0.00001:
            if BestTop3 < fit_data.history["val_T3"][i]:
                BestEpoch = i
                BestTop3 = fit_data.history["val_T3"][i]
            elif BestTop3 == 0 and (BestTop5 < fit_data.history["val_T5"][i]):
                BestEpoch = i
                BestTop5 = fit_data.history["val_T5"][i]

        if WorsTop5 >= fit_data.history["val_T5"][i]:
            WorsEpoch = i
            WorsTop5 = fit_data.history["val_T5"][i]
        elif WorsTop5 >= 0.9999:
            if WorsTop3 >= fit_data.history["val_T3"][i]:
                WorsEpoch = i
                WorsTop3 = fit_data.history["val_T3"][i]
            elif WorsTop3 == 1 and (WorsTop1 >= fit_data.history["val_T1"][i]):
                WorsEpoch = i
                WorsTop1 = fit_data.history["val_T1"][i]

    final_log_info = model.name + " results:\n\tTest-Evaluation Top 5: " + str(eval_data[1]) + ", Top 3: " + str(eval_data[2]) + ", Top 1: "  + str(eval_data[-1])
    final_log_info += "\n\tFVAs were Top 5: " + str(fit_data.history["val_T5"][-1]) + ", Top 3: " + str(fit_data.history["val_T3"][-1]) + ", Top 1: " + str(fit_data.history["val_T1"][-1])
    final_log_info += "\n\tBest Epoch was (" + str(BestEpoch+1) + ") w/ Top 5: " + str(fit_data.history["val_T5"][BestEpoch]) + ", Top 3: " + str(fit_data.history["val_T3"][BestEpoch]) + ", Top 1: "  + str(fit_data.history["val_T1"][BestEpoch])
    final_log_info += "\n\tWorst Epoch was (" + str(WorsEpoch+1) + ") w/ Top 5: " + str(fit_data.history["val_T5"][WorsEpoch]) + ", Top 3: " + str(fit_data.history["val_T3"][WorsEpoch]) + ", Top 1: "  + str(fit_data.history["val_T1"][WorsEpoch])
    final_log_info += "\n\tTraining Time was: " + str(int(fit_time)) + "s, Evaluation Time was: " + str(int(eval_time)) + "s"
    final_log_info += "\n\tNaN Check: " + str(fit_data.history["loss"][-1]) + ", " + str(fit_data.history["val_loss"][-1])
    final_log_info += "\n"
    logger.info(final_log_info)