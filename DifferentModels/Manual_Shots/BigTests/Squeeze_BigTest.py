import tensorflow as tf
import InferredActivations.Inferrer as II
import Helpers as h
import time
from keras import layers, models

logger = h.create_logger("squeeze_bigtest_dump.log")

TYPE = "shiftlu"
STATISTICS = True
assert TYPE in ["control", "pwlu", "al", "nupwlu", "shiftlu", "shiftleaky", "leaky", "prelu", "elu"]
DROPOUT = 0.5

BATCH_SIZE = 128 if TYPE == "control" else (64 if TYPE == "al" else 32)
CONCAT_AUG = True
AUGMENT_FACTOR = 0.11

EPOCHS = 15
INIT_LRATE = 0.04 #0.04 according to paper github
LRATE_SCHED = 5 #Could not locate the schedule they used
LRATE_RATIO = 0.1#`                                     `

W_DECAY = 0.0002 #0.0002 according to paper github
MOMENTUM = 0.9 #assuming SGD here (0.9 according to paper github)

IMAGE_SIZE = (224, 224)
print((
    'Beginning Squeeze Net (Fire Module) Sandbox!'
    f'\n\t{TYPE} type'
    f'\n\n\t{BATCH_SIZE} batch size'
    f'\n\t{AUGMENT_FACTOR} augment factor'
    f'\n\t{DROPOUT} drop out'
    f'\n\tConcatenate Augment? {CONCAT_AUG}'
    f'\n\n\t{EPOCHS} total epochs + Statistical Analysis Period (PWLU only)? {STATISTICS}'
    f'\n\t{INIT_LRATE} initial learning rate'
    f'\n\tLearning rate schedule of {LRATE_RATIO} ratio every {LRATE_SCHED} epochs'
    f'\n\t{W_DECAY} weight decay'
    f'\n\t{MOMENTUM} SGD momentum'
))

pwlu_v = II.PiecewiseLinearUnitV1 if TYPE == "pwlu" else II.NonUniform_PiecewiseLinearUnit
act_to_use = layers.Activation if TYPE == 'control' else (II.ActivationLinearizer if TYPE == "al" else pwlu_v)
act_arg = "relu" if TYPE != "pwlu" and TYPE != "nupwlu" else 5

if TYPE == "shiftlu":
    act_to_use = II.ShiftReLU
    act_arg = 0
elif TYPE == "shiftleaky":
    act_to_use = II.LeakyShiftReLU
    act_arg = 0
elif TYPE == "leaky":
    act_to_use = layers.LeakyReLU
    act_arg = 0.3
elif TYPE == "prelu":
    act_to_use = layers.PReLU
    act_arg = 'zeros'
elif TYPE == "elu":
    act_to_use = layers.ELU
    act_arg = 1.0

print("\nPrepping CIFAR-10 Dataset")
train_ds, val_ds = h.load_cifar10(BATCH_SIZE)

train_ds = h.prepare_dataset(train_ds, IMAGE_SIZE, augment_factor=AUGMENT_FACTOR, concatenate_augment=CONCAT_AUG)
h.report_dataset_size("Training", train_ds, BATCH_SIZE)
val_ds = h.prepare_dataset(val_ds, IMAGE_SIZE)
h.report_dataset_size("Validation", val_ds, BATCH_SIZE)

print("\nDefining the Fire Module")
class FireModule(layers.Layer):
    def __init__(self, squeeze=16, expand=64):
        super(FireModule, self).__init__()
        self.squeeze=squeeze
        self.expand=expand

    def build(self, input_shape):
        self.sLayer = models.Sequential([
            layers.Conv2D(self.squeeze, 1),
            act_to_use(act_arg),
        ])

        self.eOneLayer = models.Sequential([
            layers.Conv2D(self.expand, 1),
            act_to_use(act_arg),
        ])

        self.eThreeLayer = models.Sequential([
            layers.Conv2D(self.expand, 3, padding='same'),
            act_to_use(act_arg),
        ])

        self.sLayer.build(input_shape)
        self.eOneLayer.build(self.sLayer.compute_output_shape(input_shape))
        self.eThreeLayer.build(self.sLayer.compute_output_shape(input_shape))

    def StatisticalAnalysisToggle(self, to=None):
        assert TYPE == "pwlu" or TYPE == "nupwlu"

        self.sLayer.layers[1].StatisticalAnalysisToggle(to)
        self.eOneLayer.layers[1].StatisticalAnalysisToggle(to)
        self.eThreeLayer.layers[1].StatisticalAnalysisToggle(to)

    def call(self, input):
        x = self.sLayer(input)

        left = self.eOneLayer(x)
        right = self.eThreeLayer(x)

        return layers.concatenate([left, right], axis=3)
             
    
print("\nEstablishing Multi-GPU Training Target")
strat = tf.distribute.MirroredStrategy()
print("\tDevices to train on: {}".format(strat.num_replicas_in_sync))

with strat.scope():
    optim = tf.keras.optimizers.experimental.SGD(INIT_LRATE, MOMENTUM, weight_decay=(W_DECAY if W_DECAY > 0 else None))

    squeeze_net = models.Sequential([
         layers.Conv2D(96, 7, strides=2, input_shape=(*IMAGE_SIZE, 3)),
        layers.MaxPooling2D(3, 2),
        FireModule(16, 64),
        FireModule(16, 64),
        FireModule(32, 128),
        layers.MaxPooling2D(3, 2),
        FireModule(32, 128),
        FireModule(48, 192),
        FireModule(48, 192),
        FireModule(64, 256),
        layers.MaxPooling2D(3, 2),
        FireModule(64, 256),
        layers.Conv2D(10, 1, strides=1),
        layers.AveragePooling2D(12, 1),
        layers.Flatten(),
        layers.Activation("softmax")
    ])

    calls = [
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.LearningRateScheduler(h.lr_schedule_creator(LRATE_SCHED, LRATE_RATIO))
    ]

    metrics_to_use = [
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="T5"),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="T3"),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1, name="T1"),
    ]


print(f"\nTraining Squeeze Net in {TYPE} conditions")
squeeze_net.compile(optim, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=metrics_to_use)
if STATISTICS and (TYPE == "pwlu" or TYPE == "nupwlu"):
    print(f"\tBeginning Statistical Analysis Period")
    h.PWLU_Set_StatisticalToggle(squeeze_net, "FireModule", True)
    print(f"\t\tRunning Statistical Analysis")
    tme = time.time()
    for x, y in train_ds:
        squeeze_net.call(x)
    print(f"\t\tStatistics took {time.time() - tme}s to calculate!")
    h.PWLU_Set_StatisticalToggle(squeeze_net, "FireModule", False)

hist = squeeze_net.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=calls)

h.output_training_history(logger, hist)
h.output_validation_history(logger, hist)