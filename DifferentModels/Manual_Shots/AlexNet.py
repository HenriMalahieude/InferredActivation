#References: https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
import tensorflow as tf
import tensorflow_datasets as tfds
import time
import InferredActivations.Inferrer as II
from keras import layers

BATCH_SIZE = 128
IMAGE_SIZE = (227, 227, 3)
DROPOUT_RATE = 0.5

print("Starting AlexNet Sandbox")
print("\nLoading imagenette/320px-v2")
train_ds, val_ds = tfds.load("imagenette/320px-v2", split=['train', 'validation'], as_supervised=True, batch_size=BATCH_SIZE)
print("\t" + str(train_ds.cardinality() * BATCH_SIZE) + " training images loaded.")
print("\t" + str(val_ds.cardinality() * BATCH_SIZE) + " validation images loaded.")

resize_and_rescale = tf.keras.models.Sequential([
    layers.Resizing(IMAGE_SIZE[0], IMAGE_SIZE[1]),
    layers.Rescaling(1./255),
])

def preprocess1(x, y):
    return resize_and_rescale(x, training=True), y

base_train = train_ds.map(preprocess1)
val_data = val_ds.map(preprocess1)

data_augmentation = tf.keras.models.Sequential([
    #layers.RandomZoom((-0.5, -0.25), (-0.5, -0.25)),
    layers.RandomCrop(IMAGE_SIZE[0], IMAGE_SIZE[1]),
    #layers.RandomFlip(),
    layers.RandomContrast(factor=0.25),
    layers.RandomBrightness(factor=0.25),
    layers.Rescaling(1./255),
])

#Data Augmentation
def preprocess2(x, y):
    #x = tf.image.resize(x, (454, 454))
    #x = tf.image.central_crop(x, 0.5)
    #x = tf.image.random_crop(value=x, size=(BATCH_SIZE, 227, 227, 3))
    #x = tf.image.random_flip_left_right(x)
    #x = tf.image.random_flip_up_down(x)
    return data_augmentation(x, training=True), y

print("\nAugmenting Data")
train_data = base_train.concatenate(train_ds.map(preprocess2))
print("\tTraining Data increased to:", str(train_data.cardinality() * BATCH_SIZE))

print("\nCreating Mirrored Multi-GPU Strategy")
strat = tf.distribute.MirroredStrategy()
print("\tNumber of Devices: {}".format(strat.num_replicas_in_sync))

#Taken from Wikipedia on Alex Net, but they're picture noted that it had wrong math so... ? Let's hope this is accurate
#Used some reference at https://www.kaggle.com/code/blurredmachine/alexnet-architecture-a-complete-guide too
with strat.scope():
    alex_net = tf.keras.models.Sequential()
    alex_net.add(tf.keras.layers.Conv2D(96, 11, strides=(4, 4), input_shape=IMAGE_SIZE))
    alex_net.add(tf.keras.layers.Activation('relu'))
    alex_net.add(tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    alex_net.add(tf.keras.layers.Conv2D(256, 5, padding='same'))
    alex_net.add(tf.keras.layers.Activation('relu'))
    alex_net.add(tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    alex_net.add(tf.keras.layers.Conv2D(384, 3, padding='same'))
    alex_net.add(tf.keras.layers.Activation('relu'))
    alex_net.add(tf.keras.layers.Conv2D(384, 3, padding='same'))
    alex_net.add(tf.keras.layers.Activation('relu'))
    alex_net.add(tf.keras.layers.Conv2D(256, 3, padding='same'))
    alex_net.add(tf.keras.layers.Activation('relu'))
    alex_net.add(tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(2,2)))
    alex_net.add(tf.keras.layers.Flatten())

    alex_net.add(tf.keras.layers.Dense(4096))
    alex_net.add(tf.keras.layers.Activation('relu'))
    alex_net.add(tf.keras.layers.Dropout(DROPOUT_RATE))

    alex_net.add(tf.keras.layers.Dense(4096))
    alex_net.add(tf.keras.layers.Activation('relu'))
    alex_net.add(tf.keras.layers.Dropout(DROPOUT_RATE))

    alex_net.add(tf.keras.layers.Dense(1000))
    alex_net.add(tf.keras.layers.Activation('relu'))

    alex_net.add(tf.keras.layers.Dense(10)) #Number of Classes in our thing
    alex_net.add(tf.keras.layers.Activation('softmax'))

    metrics_to_use = [
        tf.keras.metrics.TopKCategoricalAccuracy(name='T5'), 
        tf.keras.metrics.TopKCategoricalAccuracy(name='T3', k=3), 
        tf.keras.metrics.TopKCategoricalAccuracy(name='T1', k=1)
    ]

#alex_net.summary(line_length=100)

print("\nTraining Model w/ {} of Dropout".format(DROPOUT_RATE))
alex_net.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=metrics_to_use)
alex_net.fit(x=train_data, epochs=10, validation_data=val_data)

print("\n")

#then = time.time()
#alex_net.evaluate(val_data)
#print("Took: ", str(int((time.time() - then) * 1000)) + "ms")

#Basic AlexNet (T5 Acc) -> 3.306s for 95-100% final validation accuracy (Training is ~50-60% top-5 acc)
#---
#AlexNet w/1PLU (T5 Acc) -> 3.739s for 55% f.v.a. (Training is ~40-60%)
#AlexNet w/2PLU (T5 Acc) -> 5.439s for 100% f.v.a.  (Immediately got to 100% top-5 validation accuracy) (Training is ~50-100% depending on init)
#           NOTE: Sometimes loss gets EXORBITANTLY massive at least in training, validation remains around 2.3 loss
#``          `` (T4 Acc) -> 5.428s for 100% f.v.
#           NOTE: Same thing here, training loss gets exorbitantly high (742183.3125), but validation loss is always 2.3
#``          `` (T3 Acc) -> 5.438s for 100% f.v.
#``          `` (T2 Acc) -> 5.448s for 0% f.v. (Training is ~0-40% depending on init)
#AlexNet w/3PLU (T5 Acc) -> 5.494s for 100% f.v. (Training is ~80-100%)
#AlexNet w/4PLU (T5 Acc) -> 5.563s for 100% f.v. & 4082 loss (Training is ~50-98%)
#---
#AlexNet w/1AL Init=relu (T5 Acc) -> 3.310s for 98-99% f.v. & 2.3 loss (Training is ~52-90-76%)
#AlexNet w/2AL Init=relu (T5 Acc) -> 3.317s for 100% f.v. & 2.3 loss (Training is ~50-95-80%)
#AlexNet w/3AL Init=relu (T5 Acc) -> 3.340s for 66% f.v. & 2.1 loss (max of 72% v.a.) (Training is ~55-60%)
#AlexNet w/4AL Init=relu (T5 Acc) -> 3.337s for 0% f.v. & 23.9 loss (max of 50% v.a.) (Training is ~55-60%)
#       NOTE: These replace relus bottom up, and entire model was missing 4 ReLUs

#------------------------------------------Multi-GPU
#NOTE: Replacing ReLUs top down
#Control ->  5.075s for 100% FVT5A,   0% FVT32A , but 100% T3A when trained other times (Training is T5 ~95-93%   , T3 ~48-63-61%, T2 ~29-42%   )


#3PLUs   ->  4.980s for 100% FVT2A,  -          , has been 100% T2A since Epoch 2       (Training is T5 ~53-59%   , T3 ~32-30-35%, T2 ~17-25-23%)
#. . .
#6PLUs   ->  5.181s for  47% FVT5A, 0% FVT32A   , but 97% T5 and 64% T3 at Epoch 3      (Training is T5 ~57-70-66%, T3 ~31-50%   , T2 ~18-34%   )
#. . .
#8PLUs   ->  6.000s for 100% FVT2A,  -          , was 100% T2A at Epoch 2 only          (Training is T5 ~52-55%   , T3 ~32-37-32%, T2 ~19-24-20%)


#3ALs    ->  5.000s for 100% FVT3A, 0% FVT2A    , was 100% T3A Epoch 5 only             (Training is T5 ~50-77-67%, T3 ~31-50-38%, T2 ~23-35-24%)
#. . .
#6ALs    ->  5.023s for 100% FVT2A,  -          , had 100% T3A & 98% T2A Epoch 1 only   (Training is T5 ~60-90%   , T3 ~38-67&   , T2 ~25-47%   )
#. . .
#8ALs    ->  5.117s for   0% FVT532A,-          , but 100% T5A Epochs 1, 2 & 3          (Training is T5 ~66-84%   , T3 ~34-65%   , T2 ~19-50%   )

#------------------------------------------Long Test of 50 Epochs
#NOTE: For this control, T1 is actually T2 v
#            (T5) (T3) (T2)   (TL)    (V5) (V3) (V2)   (VL)     (BE) (V5) (V3) (V2)   (VL)
#Control ->  100% 100% 090% - 2.302 | 100% 100% 100% - 2.304 || (13) 100% 100% 100% - 2.304 (NOTE: All Top 1 accuracies were 0% on second test)

#            (T5) (T3) (T1)   (TL)    (V5) (V3) (V1)   (VL)     (BE) (V5) (V3) (V1)   (VL)
#All PLUs->  088% 050% 006% - 2.302 | 100% 000% 000% - 2.304 || (13) 100% 100% 099% - 2.303
#All ALs ->  100% 100% 000% - 2.302 | 100% 100% 000% - 2.304 || (11) 100% 100% 000% - 2.305