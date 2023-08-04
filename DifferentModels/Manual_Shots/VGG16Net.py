#References: https://builtin.com/machine-learning/vgg16
#            https://datagen.tech/guides/computer-vision/vgg16/#
import tensorflow as tf
import tensorflow_datasets as tfds
import InferredActivations.Inferrer as II
import time
from keras import layers

BATCH_SIZE = 32
IMAGE_SIZE = (224, 224, 3)
DROPOUT_RATE = 0.5

print("Starting VGGNet16-C Sandbox")
print("\nLoading imagenette/320px-v2")
train_ds, val_ds = tfds.load("imagenette/320px-v2", split=['train', 'validation'], as_supervised=True, batch_size=BATCH_SIZE)
print("\t" + str(train_ds.cardinality() * BATCH_SIZE) + " training images/labels loaded.")
print("\t" + str(val_ds.cardinality() * BATCH_SIZE) + " validation images/labels loaded.")

resize_and_rescale = tf.keras.models.Sequential([
    layers.Resizing(IMAGE_SIZE[0], IMAGE_SIZE[1]),
    layers.Rescaling(1./255),
])

data_augmentation = tf.keras.models.Sequential([ # This may not be as successful as I want it to be
    layers.RandomCrop(IMAGE_SIZE[0], IMAGE_SIZE[1]),
    layers.RandomFlip(),
    layers.RandomContrast(factor=0.25),
    layers.RandomBrightness(factor=0.25),
    layers.Rescaling(1./255),
])

def preprocess1(x, y):
    return resize_and_rescale(x, training=True), y

def preprocess2(x, y):
    """x = tf.image.resize(x, (IMAGE_SIZE[0] * 2, IMAGE_SIZE[1] * 2))
    x = tf.image.central_crop(x, 0.5)
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)#"""
    return data_augmentation(x, training=True), y

print("\nAugmenting Data")
base_train = train_ds.map(preprocess1)
train_data = base_train.concatenate(train_ds.map(preprocess2))
#train_data = train_data.shuffle(train_data.cardinality()) #This just takes forever

val_data = val_ds.map(preprocess1)
print("\tTraining Data increased to:", str(train_data.cardinality() * BATCH_SIZE))

for train_x, train_y in train_data:
    print("\tShape is TX:", train_x.shape, "TY:", train_y.shape, "")
    #print(train_x[0][0])
    break

print("\nCreating Mirrored Multi-GPU Strategy")
strat = tf.distribute.MirroredStrategy()
print('\tNumber of devices: {}'.format(strat.num_replicas_in_sync))

print("\nCreating Model in Strategy")
with strat.scope():
    vggnet = tf.keras.models.Sequential([
        layers.Conv2D(64, 3, input_shape=IMAGE_SIZE, padding="same"),
        II.ActivationLinearizer(initial_eq='relu'),#layers.Activation("relu"),
        layers.Conv2D(64, 3, input_shape=IMAGE_SIZE, padding="same"),
        II.ActivationLinearizer(initial_eq='relu'),#layers.Activation("relu"),

        layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
        layers.Conv2D(filters=128, kernel_size=(3,3), padding="same"),
        II.ActivationLinearizer(initial_eq='relu'),#layers.Activation("relu"),
        layers.Conv2D(filters=128, kernel_size=(3,3), padding="same"),
        II.ActivationLinearizer(initial_eq='relu'),#layers.Activation("relu"),

        layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
        layers.Conv2D(filters=256, kernel_size=(3,3), padding="same"),
        II.ActivationLinearizer(initial_eq='relu'),#layers.Activation("relu"),
        layers.Conv2D(filters=256, kernel_size=(3,3), padding="same"),
        II.ActivationLinearizer(initial_eq='relu'),#layers.Activation("relu"),
        layers.Conv2D(filters=256, kernel_size=(1,1), padding="same"),
        II.ActivationLinearizer(initial_eq='relu'),#layers.Activation("relu"),

        layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
        layers.Conv2D(filters=512, kernel_size=(3,3), padding="same"),
        II.ActivationLinearizer(initial_eq='relu'),#layers.Activation("relu"),
        layers.Conv2D(filters=512, kernel_size=(3,3), padding="same"),
        II.ActivationLinearizer(initial_eq='relu'),#layers.Activation("relu"),
        layers.Conv2D(filters=512, kernel_size=(1,1), padding="same"),
        II.ActivationLinearizer(initial_eq='relu'),#layers.Activation("relu"),

        layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
        layers.Conv2D(filters=512, kernel_size=(3,3), padding="same"),
        II.ActivationLinearizer(initial_eq='relu'),#layers.Activation("relu"),
        layers.Conv2D(filters=512, kernel_size=(3,3), padding="same"),
        II.ActivationLinearizer(initial_eq='relu'),#layers.Activation("relu"),
        layers.Conv2D(filters=512, kernel_size=(1,1), padding="same"),
        II.ActivationLinearizer(initial_eq='relu'),#layers.Activation("relu"),

        layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
        layers.Flatten(),
        layers.Dense(4096),
        II.ActivationLinearizer(initial_eq='relu'),#layers.Activation("relu"),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(4096),
        II.ActivationLinearizer(initial_eq='relu'),#layers.Activation("relu"),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(10, activation='softmax'),
    ])

    metrics_to_use = [tf.keras.metrics.TopKCategoricalAccuracy(name='T5A'), tf.keras.metrics.TopKCategoricalAccuracy(name='T3A', k=3), tf.keras.metrics.TopKCategoricalAccuracy(name='T2A', k=2)]

#vggnet.summary()

print("\nTraining Model w/" + str(DROPOUT_RATE) +" of Dropout")
#checkpoint_saving = tf.keras.callbacks.ModelCheckpoint(verbose=1, )
#early_stop = tf.keras.callbacks.EarlyStopping(monitor='top_k_categorical_accuracy', min_delta=0.01, patience=5, verbose=1, mode='auto')

#vggnet = utils.multi_gpu_model(vggnet, gpus=2)

vggnet.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=metrics_to_use)
vggnet.fit(x=train_data, epochs=5, validation_data=val_data)

print("\n")

then = time.time()
vggnet.evaluate(val_data)

print("Took: " + str(int((time.time() - then) * 1000)) + "ms")

#Base VGGNET -> 5.938s for exactly 0 or 100% FVT3A [Final Validation Top-3 Acc] (Training is ~30-70%)

#VGGnet w/1PLU @ Bot -> 31.896s for exactly 0% FVT3A (Training is ~40-50%)
#           NOTE: Training a singular epoch got dramatically longer, from ~100s to ~400-500s (Because of batch size reduction)
#           NOTE: Needed to reduce size of batches for the memory to not have any issues for PLU
#VGGNet w/2PLU @ Bot -> 61.477s for exactly 0% FVT3A (Training is ~35-40%)
#           NOTE: Had to reduce batch size again because of memory constraint, so that increased training time from 500s to literally past 12min

#VGGNet w/1PLU @ Top -> 6.973s for 100% x3 FVT4A (Training is ~40-50%)
#VGGNet w/3PLU @ Top -> 6.818s for 0% x3 FVT4A (With 99.85-100% success sometimes) (Training is ~40-50%)

#---------------------------------------------------------------------------Multi-GPU
#`    ` w/4PLU B: 128 @ Top  -> 5.890s for 100% x4 FVT5A, 31% FVT3A (100% x2 other times) (Training is ~50-53% T5A, ~30-29% T3A)
#`    ` w/6PLU B: 128 @ Top -> 7.914s for 100% FVT5A, 0% FVT32A, but 100% T2A at Epoch 4 (Training is ~52% T5, ~31% T3, ~22-20% T2)
#`    ` w/7PLU B: 128 @ Top -> 10.180s for 100% FVT5A, 0% FVT32A, but 100% T2A at Epoch 3 (Training is ~50-53% T5, ~31% T3, ~19-21%)


#Control VGGNet B:128 ->  5.359s for 100% FVT5A. 0% FVT32A, but 100% T5A at Epoch 1-3&5 (Training is ~69-86-80% T5, ~36-55-48% T3, ~25-36-26% T2)


#VGGNet w/2PLU B: 128 ->  5.629s for   0% FVT5A,            but 100% T2A at Epoch 3     (Training is ~50?%      T5, ~30?%      T3, ~20?%      T2)
#. . .
#`    ` w/5PLU B: 128 ->  6.296s for 100% FVT3A, 0% FVT2A,  but 100% T2A at Epoch 2-4   (Training is ~50-52%    T5, ~32%       T3, ~20-21%    T2) 
#. . .
#`    ` w/8PLU B: 128 -> 11.885s for 100% FVT5A, 0% FVT32A, but 100% T3A at Epoch 3     (Training is ~50-67%    T5, ~31-41%    T3, ~21-25%    T2)
#. . .
#`    ` w/11PLU B: 64 -> 22.617s for  92% FVT5A, 0% FVT32A, but 100% T2A at Epoch 2     (Training is ~51-64%    T5, ~31-35%    T3, ~20-21%    T2)
#. . .
#`    ` w/13PLU B: 32 -> 38.418s for   0% FVT5A,            but 100% T2A at Epoch 3     (Training is ~52-74%    T5, ~31-40%    T3, ~20-26-21% T2)
#. . .
#`    ` w/15PLU B: 16 -> 74.436s for   0% FVT5A,            but 100% T3A at Epoch 1     (Training is ~47-61-53% T5, ~32-40%    T3, ~21-30%    T2)


#VGGNet w/2AL  B: 128 ->  5.311s for 100% FVT3A, 0% FVT2A,  but 100% T2A at Epoch 1     (Training is ~63-57%    T5, ~36-30%    T3, ~20-19%    T2)
#. . .
#`    ` w/5AL  B: 128 ->  6.414s for 100% FVT2A,            but 100% T3A at Epoch 1/4   (Training is ~52-57%    T5, ~32-37%    T3, ~21-26%    T2)
#. . .
#`    ` w/8AL  B: 128 ->  6.317s for 100% FVT3A,                                        (Training is ~58-56%    T5, ~33-37-36% T3, ~20-25%    T2)
#. . .
#`    ` w/11AL B:  64 ->  9.482s for   0% FVT532A,          but 100% T5A at Epoch 3     (Training is ~58-50-56% T5, ~36-31-36% T3, ~24-20-24% T2)
#. . .
#`    ` w/13AL B:  32 -> 12.054s for 100% FVT5A, 0% FVT32A,                             (Training is ~51-55-53% T5, ~33-35-32% T3, ~23-21%    T2)
#. . .
#`    ` w/15AL B:  32 -> 22.431s for 100% FVT5A, 0% FVT32A,  but 100% T3A at Epoch 1    (Training is ~54%       T5, ~33%       T3, ~22-25%    T2)