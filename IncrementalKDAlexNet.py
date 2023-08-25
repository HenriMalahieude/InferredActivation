import tensorflow as tf
import tensorflow_datasets as tfds
import InferredActivations.Inferrer as II
from KDLib import Distiller
from keras import models, layers

IMAGE_SIZE = (227, 227, 3)
BATCH_SIZE = 128
DROPOUT_RATE = 0.5
AUGMENT_DATA = True
CONCATENATE_AUGMENT = True

DISTILL_KWARGS = {
    "optimizer": "adam",
    "student_loss_fn": tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM),
    "distillation_loss_fn": tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.SUM),
    "alpha": 0.1,
    "temperature": 5,
}

print("Incremental Knowledge Distillation for AlexNet")
print("\tBatched in {}\n\tDropout Rate of {}".format(BATCH_SIZE, DROPOUT_RATE))

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

print("\tPreprocessing Datasets")
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

print("\nEstablishing Multi-GPU Target")
strat = tf.distribute.MirroredStrategy()
print("\t\tAvailable Devices: {}".format(strat.num_replicas_in_sync))

print("\nTeacher Model")
with strat.scope():
    metrics_to_use = [
        tf.keras.metrics.TopKCategoricalAccuracy(name='T5'), 
        tf.keras.metrics.TopKCategoricalAccuracy(name='T3', k=3), 
        tf.keras.metrics.TopKCategoricalAccuracy(name='T1', k=1),
        tf.keras.metrics.SparseCategoricalAccuracy(name='SCAcc')
    ]

    teacher_base = models.Sequential([
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
    ], name = "teacher_base")

print('\tTraining Teacher')
teacher_base.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=metrics_to_use)
teacher_base.fit(train_data, epochs=4, validation_data=val_data)

print("\nCreating Teacher and Student Squad")
with strat.scope():
    teacher_base.pop() #Remove Softmax layer
    teacher_top = models.clone_model(teacher_base)

    for i in range(9):
        teacher_base.pop()

    teacher_middle = models.clone_model(teacher_base)

    for i in range(7):
        teacher_base.pop()

    teacher_bottom = models.clone_model(teacher_base)

    print("\tBuilding First Stage on Student")
    student_one = models.Sequential([
        layers.Conv2D(96, 11, strides=(4, 4), input_shape=IMAGE_SIZE),
        II.ActivationLinearizer(),
        layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2)),

        layers.Conv2D(256, 5, padding='same'),
        II.ActivationLinearizer(),
        layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2)),
    ])

    print("\tBuilding Second Stage")
    student_two = models.Sequential([
        student_one,

        layers.Conv2D(384, 3, padding='same'),
        II.ActivationLinearizer(),
        layers.Conv2D(384, 3, padding='same'),
        II.ActivationLinearizer(),
        layers.Conv2D(256, 3, padding='same'),
        II.ActivationLinearizer(),
        layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2))
    ])
    
    print("\tBuilding Third Stage")
    student_three = models.Sequential([
        student_two,

        layers.Dense(4096),
        II.ActivationLinearizer(),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(4096),
        II.ActivationLinearizer(),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(1000),
        II.ActivationLinearizer(),
        layers.Dense(10)
    ])

    stage_one = Distiller(student=student_one, teacher=teacher_bottom)
    stage_one.compile(metrics=[], **DISTILL_KWARGS)

    stage_two = Distiller(student=student_two, teacher=teacher_middle)
    stage_two.compile(metrics=[], **DISTILL_KWARGS)

    stage_three = Distiller(student=student_three, teacher=teacher_top)
    stage_three.compile(
        metrics=[], 
        optimizer="adam", 
        student_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM),
        distillation_loss_fn=tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.SUM),
        alpha = 0.1,
        temperature=5
    )

print("\nTraining First Stage of Student")
stage_one.fit(train_data, epochs=4)
#stage_one.evaluate(val_data)

print("\nTraining Second Stage of Student")
stage_two.fit(train_data, epochs=4)
#stage_two.evaluate(val_data)

print("\nTraining Third Stage of Student")
stage_three.fit(train_data, epochs=4)
stage_three.evaluate()

print("\nIndependent Training")
student_three.add(layers.Activation("softmax"))
student_three.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=metrics_to_use)
student_three.fit(train_data, epochs=2, validation_data=val_data)