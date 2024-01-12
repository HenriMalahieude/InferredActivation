import time
import tensorflow as tf
import Helpers as h
import InferredActivations.Inferrer as II
from keras import layers, models

#Setting up Logger
logger = h.create_logger("squeeze_excitation_net_dump.log")

TYPE = "pwlu"
STATISTICS = True #PWLU only
assert TYPE in ["control", "pwlu", "al", "nupwlu"]
AUGMENT_FACTOR = 0.1
CONCAT_AUG = True

BATCH_SIZE = 128 if TYPE == "control" else (64 if TYPE == "al" else 32)
EPOCHS = 15
REDUCTION_RATIO = 16 #16 as paper says

INIT_LRATE = 0.01 #0.6 initial by paper
LRATE_SCHED = 30 #30 epochs by paper
LRATE_RATIO = 0.1 #0.1 ratio by paper

W_DECAY = 0.0 #0 by paper
MOMENTUM = 0.9 #0.9 by paper

IMAGE_SIZE = (224, 224)

print((
	'Beginning squeeze excitation net testing!'
	f'\n\t{TYPE} type'
	f'\n\n\t{BATCH_SIZE} batch size'
	f'\n\t{AUGMENT_FACTOR} augment factor'
	f'\n\tConcatenate Augment? {CONCAT_AUG}'
	f'\n\n\t{EPOCHS} total epochs + Statistical Analysis Period (PWLU only)? {STATISTICS}'
    f'\n\t{REDUCTION_RATIO} Reduction Ratio'
	f'\n\t{INIT_LRATE} initial learning rate'
	f'\n\tLearning rate schedule of {LRATE_RATIO} every {LRATE_SCHED} epochs'
	f'\n\t{W_DECAY} weight decay'
	f'\n\t{MOMENTUM} SGD momentum'
))

pwlu_v = II.PiecewiseLinearUnitV1 if TYPE == "pwlu" else II.NonUniform_PiecewiseLinearUnit
act_to_use = layers.Activation if TYPE == 'control' else (II.ActivationLinearizer if TYPE == "al" else pwlu_v)
act_arg1 = "relu" if TYPE != "pwlu" and TYPE != "nupwlu" else 5
act_arg2 = "sigmoid" if TYPE != "pwlu" and TYPE != "nupwlu" else 5

print("\nPrepping CIFAR-10 Dataset")
train_ds, val_ds = h.load_cifar10(BATCH_SIZE)

train_ds = h.prepare_dataset(train_ds, IMAGE_SIZE, augment_factor=AUGMENT_FACTOR, concatenate_augment=CONCAT_AUG)
h.report_dataset_size("Training", train_ds, BATCH_SIZE)
val_ds = h.prepare_dataset(val_ds, IMAGE_SIZE)
h.report_dataset_size("Validation", val_ds, BATCH_SIZE)

print("\nDefining Layers")
#Taken from https://github.com/taki0112/SENet-Tensorflow/tree/master as reference
class SE_ResidualBlock(layers.Layer):
    def __init__(self, 
                 filters=(64,64,256),
                 stride=1,):
        super(SE_ResidualBlock, self).__init__()
        self.filters = filters
        self.stride = stride
    
    def build(self, input_shape):
        f1, f2, f3 = self.filters

        self.normal_pass = models.Sequential([
            layers.Conv2D(f1, 1, strides=self.stride),
            layers.BatchNormalization(),
            act_to_use(act_arg1),

            layers.Conv2D(f2, 3, padding='same'),
            layers.BatchNormalization(),
            act_to_use(act_arg1),

            layers.Conv2D(f3, 1),
            layers.BatchNormalization(),
        ])

        self.excitation = models.Sequential([
            layers.GlobalAveragePooling2D(),

            layers.Dense(units=(f3/REDUCTION_RATIO), use_bias=False), #Rest of this is the "excitation layer"
            act_to_use(act_arg1),
            layers.Dense(units=f3, use_bias=False),
            act_to_use(act_arg2),
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

        self.final_pass = act_to_use(act_arg1)

    def call(self, input):
        y = self.normal_pass(input)
        #print(y.shape)
        yy = self.excitation(y)#bc this works better sighhhhh
        #print(yy.shape)

        return self.final_pass((y * yy) + self.id_pass(input))
    
    def StatisticalAnalysisToggle(self, to):
        assert TYPE == "pwlu" or TYPE == "nupwlu"

        self.normal_pass.layers[2].StatisticalAnalysisToggle(to)
        self.normal_pass.layers[5].StatisticalAnalysisToggle(to)
        self.excitation.layers[2].StatisticalAnalysisToggle(to)
        self.excitation.layers[4].StatisticalAnalysisToggle(to)

print("\nConstructing Model")
strat = tf.distribute.MirroredStrategy()
print("\t{} Available Devices".format(strat.num_replicas_in_sync))

with strat.scope():
    optim = tf.keras.optimizers.experimental.SGD(INIT_LRATE, MOMENTUM, weight_decay=(W_DECAY if W_DECAY > 0 else None))

    se_net = models.Sequential([
        layers.Conv2D(64, 7, strides=2, input_shape=(*IMAGE_SIZE, 3)),
        layers.BatchNormalization(),
        act_to_use(act_arg1),
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
        act_to_use(act_arg1),

        layers.Dense(128),
        act_to_use(act_arg1),

        layers.Dense(10, activation='softmax')
	])

    metrics_to_use = [
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="T5"),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="T3"),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1, name="T1")
    ]
    
    calls = [
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.LearningRateScheduler(h.lr_schedule_creator(LRATE_SCHED, LRATE_RATIO))
    ]

print("\nStarting training")
se_net.compile(optim, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=metrics_to_use)
if (TYPE == "pwlu" or TYPE == "nupwlu") and STATISTICS:
    h.PWLU_Set_StatisticalToggle(se_net, "SE_ResidualBlock", True)
    print(f"\t\tBeginning Statistical Analysis!")
    tme = time.time()
    for x, y in train_ds:
        se_net.call(x)
    print(f"\t\tStatistical Analysis took {time.time() - tme} seconds.")
    h.PWLU_Set_StatisticalToggle(se_net, "SE_ResidualBlock", False)

hist = se_net.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=calls)

h.output_training_history(logger, hist)
h.output_validation_history(logger, hist)