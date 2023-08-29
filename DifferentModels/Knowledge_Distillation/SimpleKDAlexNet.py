import logging
import tensorflow as tf
import InferredActivations.Inferrer as II
import tensorflow_datasets as tfds
from KDLib import Distiller
from keras import models, layers
from keras.metrics import TopKCategoricalAccuracy

#Logging Set up
logger = logging.getLogger("internal_Logger_wo__imports")
logger.setLevel(logging.DEBUG)
fileHandle = logging.FileHandler(filename="kd_alexnet_testing_suite.log")

formatter = logging.Formatter(fmt='%(message)s')
fileHandle.setFormatter(formatter)

logger.addHandler(fileHandle)

BATCH_SIZE = 128
IMAGE_SIZE = (227, 227, 3)
DROPOUT_RATE = 0.5
AUGMENT_DATA = True
CONCATENATE_AUGMENT = True

print("Starting AlexNet Knowledge Distillation")
print("\t{} Epochs\n\tBatched in {}\n\tDropout Rate of {}".format(-1, BATCH_SIZE, DROPOUT_RATE))

print("\nLoading imagenette/320px-v2")
train_ds, val_ds = tfds.load("imagenette/320px-v2", split=['train', 'validation'], as_supervised=True, batch_size=BATCH_SIZE)

resize_and_rescale = models.Sequential([
    layers.Resizing(IMAGE_SIZE[0], IMAGE_SIZE[1]),
    layers.Rescaling(1./255),
])

data_augmentation = models.Sequential([
    layers.RandomCrop(IMAGE_SIZE[0], IMAGE_SIZE[1]),
    layers.RandomFlip(),
    layers.RandomContrast(factor=0.25),
    layers.RandomBrightness(factor=0.25),
    resize_and_rescale
])

def preprocess1(x, y):
    return resize_and_rescale(x, training=True), y

def preprocess2(x, y):
    return data_augmentation(x, training=True), y
"""
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
"""

print("\tPreprocessing Datasets")
train_dataset = train_ds.map(preprocess1)
val_dataset = val_ds.map(preprocess1)

if AUGMENT_DATA:
    if CONCATENATE_AUGMENT:
        print("\tConcatenating Training with anAugment")
        train_dataset = train_dataset.concatenate(train_ds.map(preprocess2))
    else:
        print("\nReplacing Training with Augment")
        train_dataset = train_ds.map(preprocess2)

print("\nEstablishing Multi-GPU Target")
strat = tf.distribute.MirroredStrategy()
print("\t\tAvailable Devices: {}".format(strat.num_replicas_in_sync))

with strat.scope():
    metrics_to_use = [
        tf.keras.metrics.TopKCategoricalAccuracy(name='T5'), 
        tf.keras.metrics.TopKCategoricalAccuracy(name='T3', k=3), 
        tf.keras.metrics.TopKCategoricalAccuracy(name='T1', k=1),
        tf.keras.metrics.SparseCategoricalAccuracy(name='SCAcc')
    ]

    teacher_m = models.Sequential([
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
    ], name = "teacher")

    student_m = models.Sequential([
        layers.Conv2D(96, 11, strides=(4, 4), input_shape=IMAGE_SIZE),
        II.ActivationLinearizer(),#layers.Activation('relu'),
        layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2)),

        layers.Conv2D(256, 5, padding='same'),
        II.ActivationLinearizer(),#layers.Activation('relu'),
        layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2)),

        layers.Conv2D(384, 3, padding='same'),
        II.ActivationLinearizer(),#layers.Activation('relu'),
        layers.Conv2D(384, 3, padding='same'),
        II.ActivationLinearizer(),#layers.Activation('relu'),
        layers.Conv2D(256, 3, padding='same'),
        II.ActivationLinearizer(),#layers.Activation('relu'),
        layers.AveragePooling2D(pool_size=(3, 3), strides=(2,2)),
        layers.Flatten(),

        layers.Dense(4096),
        II.ActivationLinearizer(),#layers.Activation('relu'),

        layers.Dropout(DROPOUT_RATE),
        layers.Dense(4096),
        II.ActivationLinearizer(),#layers.Activation('relu'),

        layers.Dropout(DROPOUT_RATE),
        layers.Dense(1000),
        II.ActivationLinearizer(),#layers.Activation('relu'),
        
        layers.Dense(10),
        #layers.Activation('softmax'),
    ], name = "student")

#teacher_m.summary(line_length=80)
#student_m.summary(line_length=80)

print("\nTraining the Teacher")
teacher_m.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=metrics_to_use)
teacher_m.fit(train_dataset, epochs=6, validation_data=val_dataset)
teacher_m.pop() #Remove the top "softmax" layer so now it's just logits

print("\nEntering the Classroom with Student")
with strat.scope():
    classroom = Distiller(student=student_m, teacher=teacher_m)
    classroom.compile(
        optimizer="adam",
        metrics=metrics_to_use,
        student_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM),
        distillation_loss_fn=tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.SUM),
        alpha=0.1,
        temperature=5
    )

classroom.fit(train_dataset, epochs=5)

print("\nEvaluating Classroom:")
classroom.evaluate(val_dataset)

student_m.add(layers.Activation("softmax"))
student_m.compile(optimizer="adam", loss = tf.keras.losses.SparseCategoricalCrossentropy(), metrics=metrics_to_use)

print("\nSoftmax Evaluation of Student:")
student_m.evaluate(val_dataset)

#Maybe the student now needs to be left to their own devices?
print("\nIndependent Study by Student")
student_m.fit(train_dataset, epochs=2, validation_data=val_dataset)

#Got Student and Teacher to both get 100% T5, and no difference in loss
#However student only matches T5 performance, even if Teacher has T3 Performance
#Never mind sometimes it will almost match the T5, and lag behind the T3. But independent training shortly afterwards (1 epoch) can boost the T1 training to 87%