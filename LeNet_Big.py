import tensorflow as tf
import InferredActivations.Inferrer as II
import Helpers as h
from keras import layers, models

logger = h.create_logger("lenet_big_dump.log")

TYPE = "control"
assert TYPE in ["control", "pwlu", "al"]
AUGMENT_FACTOR = 0.1
CONCAT_AUG = True

BATCH_SIZE = 128 #if TYPE == "control" else 64#(64 if TYPE == "al" else 32)
EPOCHS = 15
INIT_LRATE = 0.01
LRATE_SCHED = 30
LRATE_RATIO = 0.1

W_DECAY = 0.0
MOMENTUM = 0.9

IMAGE_SIZE = (128, 128)

print((
	'Beginning efficient net v1-b0 testing!'
	f'\n\t{TYPE} type'
	#f'\n\t{BOTTLENECK_RATIO} bottleneck ratio'
	#f'\n\t{SHUFFLE_BLOCKS} shuffle blocks'
	#f'\n\t{SCALE_FACTOR} scale factor'
	f'\n\n\t{BATCH_SIZE} batch size'
	f'\n\t{AUGMENT_FACTOR} augment factor'
	f'\n\tConcatenate Augment? {CONCAT_AUG}'
	f'\n\n\t{EPOCHS} total epochs'
	f'\n\t{INIT_LRATE} initial learning rate'
	f'\n\tLearning rate schedule of {LRATE_RATIO} ratio every {LRATE_SCHED} epochs'
	f'\n\t{W_DECAY} weight decay'
	f'\n\t{MOMENTUM} SGD momentum'
))

act_to_use = layers.Activation if TYPE == 'control' else (II.ActivationLinearizer if TYPE == "al" else II.PiecewiseLinearUnitV1)
act_arg = "relu" if TYPE != "pwlu" else 20

print("\nPrepping CIFAR-10 Dataset")
train_ds, val_ds = h.load_cifar10(BATCH_SIZE)

train_ds = h.prepare_dataset(train_ds, IMAGE_SIZE, augment_factor=AUGMENT_FACTOR, concatenate_augment=CONCAT_AUG)
h.report_dataset_size("Training", train_ds, BATCH_SIZE)
val_ds = h.prepare_dataset(val_ds, IMAGE_SIZE)
h.report_dataset_size("Validation", val_ds, BATCH_SIZE)

print(f"\nDefining {TYPE} type LeNet")
strat = tf.distribute.MirroredStrategy()
print(f"\tUsing {strat.num_replicas_in_sync} GPUs")

with strat.scope():
	optim = tf.keras.optimizers.experimental.SGD(INIT_LRATE, MOMENTUM, weight_decay=(W_DECAY if W_DECAY > 0 else None))

	lesnet = models.Sequential([
        layers.Conv2D(6, 5, input_shape=(*IMAGE_SIZE, 3)),
        act_to_use(act_arg),

        layers.AveragePooling2D(2),
        act_to_use(act_arg),

        layers.Conv2D(16, 5),
        act_to_use(act_arg),

        layers.AveragePooling2D(2),
        act_to_use(act_arg),

        layers.Conv2D(120, 5),
        act_to_use(act_arg),

        layers.Flatten(),
        layers.Dense(84),
        act_to_use(act_arg),

        layers.Dense(10, activation="softmax"),
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

lesnet.summary()

"""print(f"\nStarting {TYPE} training")
lesnet.compile(optim, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=metrics_to_use)
hist = lesnet.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=calls)

h.output_training_history(logger, hist)
h.output_validation_history(logger, hist)#"""