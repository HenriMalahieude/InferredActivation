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
fileHandle = logging.FileHandler(filename="squeeze_testing_suite.log")

formatter = logging.Formatter(fmt='%(message)s')
fileHandle.setFormatter(formatter)

logger.addHandler(fileHandle)

TOTAL_EPOCHS = 1
BATCH_SIZE = 8
IMAGE_SIZE = (224, 224, 3)
AUGMENT_DATA = True
CONCATENATE_AUGMENT = False

print("Starting Squeeze Net Auto Tester")
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

print("\nModel Definitions")
@tf.keras.saving.register_keras_serializable()
class ReluActivation(layers.Layer): #because home boy tfmot complainin'
    def __init__(self):
        super(ReluActivation, self).__init__()

    def get_config(self):
        return super().get_config()
    
    @classmethod
    def from_config(cls, config):
        return ReluActivation()

    def call(self, input):
        return tf.nn.relu(input)

@tf.keras.saving.register_keras_serializable()
class SoftmaxActivation(layers.Layer):
    def __init__(self):
        super(SoftmaxActivation, self).__init__()

    def get_config(self):
        return super().get_config()
    
    @classmethod
    def from_config(cls, config):
        return SoftmaxActivation()
    
    def call(self, input):
        return tf.nn.softmax(input)

print("\tDefining Fire Module")
@tf.keras.saving.register_keras_serializable()
class FireModule(layers.Layer):
    def __init__(self, squeeze=16, expand=64 , acode_1=0, acode_2=0, acode_3=0, quantized=False):
        super(FireModule, self).__init__()
        self.squeeze=squeeze
        self.expand=expand

        self.activation_code_1 = acode_1 #0 for normal, 1 for PLU, 2 for AL
        self.activation_code_2 = acode_2
        self.activation_code_3 = acode_3

        self.quantized = quantized

    def get_config(self):
        base_config = super().get_config()
        config = {
            "squeeze": self.squeeze,
            "expand": self.expand,
            "activation_code_1": self.activation_code_1,
            "activation_code_2": self.activation_code_2,
            "activation_code_3": self.activation_code_3,
            "quantized": self.quantized
        }

        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return FireModule(config["squeeze"], config["expand"], config["activation_code_1"], config["activation_code_2"], config["activation_code_3"], config["quantized"])

    def build(self, input_shape):
        activ3 = activ2 = activ1 = ReluActivation

        if self.activation_code_1 == 1:
            activ1 = II.PiecewiseLinearUnitV1
        elif self.activation_code_1 == 2:
            activ1 = II.ActivationLinearizer

        if self.activation_code_2 == 1:
            activ2 = II.PiecewiseLinearUnitV1
        elif self.activation_code_2 == 2:
            activ2 = II.ActivationLinearizer

        if self.activation_code_3 == 1:
            activ3 = II.PiecewiseLinearUnitV1
        elif self.activation_code_3 == 2:
            activ3 = II.ActivationLinearizer

        self.sLayer = models.Sequential([
            layers.Conv2D(self.squeeze, 1),
            activ1(),
        ])

        self.eOneLayer = models.Sequential([
            layers.Conv2D(self.expand, 1),
            activ2(),
        ])

        self.eThreeLayer = models.Sequential([
            layers.Conv2D(self.expand, 3, padding='same'),
            activ3(),
        ])

        self.sLayer.build(input_shape)
        self.eOneLayer.build(self.sLayer.compute_output_shape(input_shape))
        self.eThreeLayer.build(self.sLayer.compute_output_shape(input_shape))

        if self.quantized:
            quantize_annotate_model(self.sLayer)
            quantize_annotate_model(self.eOneLayer)
            quantize_annotate_model(self.eThreeLayer)

    def call(self, input):
        x = self.sLayer(input)

        left = self.eOneLayer(x)
        right = self.eThreeLayer(x)

        return layers.concatenate([left, right], axis=3)
    
print("\tDefining Control Architecture")
control_arch = [
    #0
    {
        "lyrFn": layers.Conv2D,
        "args": [96, 7],
        "kwargs": {"strides": 2},
    },

    #1
    {
        "lyrFn": layers.MaxPooling2D,
        "args": [3, 2],
        "kwargs": {},
    },

    #2
    {
        "lyrFn": FireModule,
        "args": [],
        "kwargs": {},
    },

    #3
    {
        "lyrFn": FireModule,
        "args": [],
        "kwargs": {},
    },

    #4
    {
        "lyrFn": FireModule,
        "args": [32, 128],
        "kwargs": {},
    },

    #5
    {
        "lyrFn": layers.MaxPooling2D,
        "args": [3, 2],
        "kwargs": {},
    },

    #6
    {
        "lyrFn": FireModule,
        "args": [32, 128],
        "kwargs": {},
    },

    #7
    {
        "lyrFn": FireModule,
        "args": [48, 192],
        "kwargs": {},
    },

    #8
    {
        "lyrFn": FireModule,
        "args": [48, 192],
        "kwargs": {},
    },

    #9
    {
        "lyrFn": FireModule,
        "args": [64, 256],
        "kwargs": {},
    },

    #10
    {
        "lyrFn": layers.MaxPooling2D,
        "args": [3, 2],
        "kwargs": {},
    },

    #11
    {
        "lyrFn": FireModule,
        "args": [64, 256],
        "kwargs": {},
    },

    #12
    {
        "lyrFn": layers.Conv2D,
        "args": [1000, 1, 1],
        "kwargs": {},
    },

    #13
    {
        "lyrFn": layers.AveragePooling2D,
        "args": [12, 1],
        "kwargs": {},
    },

    #14
    {
        "lyrFn": ReluActivation,
        "args": [],
        "kwargs": {},
    },

    #15
    {
        "lyrFn": layers.Flatten,
        "args": [],
        "kwargs": {},
    },

    #16
    {
        "lyrFn": layers.Dense,
        "args": [10],
        "kwargs": {},
    },

    #17
    {
        "lyrFn": SoftmaxActivation,
        "args": [],
        "kwargs": {},
    },
]

#Primarily Notes for myself
raw_activations_normal_indices =  [14]
raw_activations_softmax_indices = [17]
fire_module_indices = [2, 3, 4, 6, 7, 8, 9, 11]

print("\nDefining Buildable Models")
buildables = [
    {
        "name": "Control-Squeeze",
        "quantized": False,
        "lyr_indices": [],
        #"lyr_replacement": layers.Activation,
        #"lyr_args": ["relu"],
        #"lyr_kwargs": {},
        "kwarg_indices": [],
        #"kwarg_replacement": {},
    },

    {
        "name": "Control-Squeeze-Quantized",
        "quantized": True,
        "lyr_indices": [],
        #"lyr_replacement": layers.Activation,
        #"lyr_args": ["relu"],
        #"lyr_kwargs": {},
        "kwarg_indices": [],
        #"kwarg_replacement": {},
    },

    {
        "name": "FullPLUs",
        "quantized": False,
        "lyr_indices": [14],
        "lyr_replacement": II.PiecewiseLinearUnitV1,
        "lyr_args": [],
        "lyr_kwargs": {},
        "kwarg_indices": [2, 3, 4, 6, 7, 8, 9, 11],
        "kwarg_replacement": {"acode_1": 1, "acode_2": 1, "acode_3": 1},
    },

    {
        "name": "FullPLUs-Quantized",
        "quantized": True,
        "lyr_indices": [14],
        "lyr_replacement": II.PiecewiseLinearUnitV1,
        "lyr_args": [],
        "lyr_kwargs": {},
        "kwarg_indices": [2, 3, 4, 6, 7, 8, 9, 11],
        "kwarg_replacement": {"acode_1": 1, "acode_2": 1, "acode_3": 1, "quantized": True},
    },

    {
        "name": "FullALs",
        "quantized": False,
        "lyr_indices": [14],
        "lyr_replacement": II.ActivationLinearizer,
        "lyr_args": ["relu"],
        "lyr_kwargs": {},
        "kwarg_indices": [2, 3, 4, 6, 7, 8, 9, 11],
        "kwarg_replacement": {"acode_1": 2, "acode_2": 2, "acode_3": 2},
    },

    {
        "name": "FullALs-Quantized",
        "quantized": True,
        "lyr_indices": [14],
        "lyr_replacement": II.ActivationLinearizer,
        "lyr_args": ["relu"],
        "lyr_kwargs": {},
        "kwarg_indices": [2, 3, 4, 6, 7, 8, 9, 11],
        "kwarg_replacement": {"acode_1": 2, "acode_2": 2, "acode_3": 2, "quantized": True},
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
        print("\t\t" + k["name"])
        new_model = models.Sequential(name=k["name"])

        for i in range(len(control_arch)):
            lyrFn = control_arch[i]["lyrFn"]
            args = control_arch[i]["args"]
            kwargs = control_arch[i]["kwargs"]

            if i in k["lyr_indices"]: #Replace that layer
                lyrFn = k["lyr_replacement"]
                args = k["lyr_args"]
                kwargs = k["lyr_kwargs"]
            elif i in k["kwarg_indices"]:
                kwargs = k["kwarg_replacement"]

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


def formatPercentage(float):
    return int(float * 10000) / 100

print("\nStarting Testing Suite")
start_at = 1 #for resuming when crashed, or annoyingly disconnected
for i in range(len(models_to_test)-start_at):
    model = models_to_test[i+start_at]
    print("\tTraining " + model.name)
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

#Why won't you actually start training?