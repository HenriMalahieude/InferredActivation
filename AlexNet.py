import tensorflow as tf
import tensorflow_datasets as tfds
import time
import InferredActivations.Inferrer as II

BATCH_SIZE = 64

train_ds, val_ds = tfds.load("imagenette/320px-v2", split=['train', 'validation'], as_supervised=True, batch_size=BATCH_SIZE)

def preprocess1(x, y):
    x = tf.image.resize(x, (227, 227))
    return x, y

base_train = train_ds.map(preprocess1)
val_data = val_ds.map(preprocess1)

#Data Augmentation
def preprocess2(x, y):
    x = tf.image.resize(x, (454, 454))
    x = tf.image.central_crop(x, 0.5)
    #x = tf.image.random_crop(value=x, size=(BATCH_SIZE, 227, 227, 3))
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    return x, y

train_data = base_train.concatenate(train_ds.map(preprocess2))

"""
for train_x, train_y in train_data:
    print(train_x.shape, train_y.shape)
    #print(train_y)
    break

min_x, min_y = 10000, 10000
max_x, max_y = 0, 0
Batch_size = 0
Color_size = 3
for x_train, y_train in train_data:
    B, X, Y, C = x_train.shape
    Batch_size = B
    Color_size = C
    if min_x * min_y > X * Y:
        min_x = X
        min_y = Y
    elif X * Y > max_x * max_y:
        max_x = X
        max_y = Y

print("Maximum seen: ", max_x, " x ", max_y)
print("Minimum seen: ", min_x, " x ", min_y)
print(Batch_size, max_x, max_y, Color_size)
#"""

DROPOUT_RATE = 0.5

#Taken from Wikipedia on Alex Net, but they're picture noted that it had wrong math so... ? Let's hope this is accurate
alex_net = tf.keras.models.Sequential()
alex_net.add(tf.keras.layers.Conv2D(96, 11, strides=(4, 4), input_shape=(227, 227, 3)))
alex_net.add(II.ActivationLinearizer(initial_eq='relu'))#tf.keras.layers.Activation('relu'))
alex_net.add(tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(2,2)))

alex_net.add(tf.keras.layers.Conv2D(256, 5, padding='same'))
alex_net.add(II.ActivationLinearizer(initial_eq='relu'))#tf.keras.layers.Activation('relu'))
alex_net.add(tf.keras.layers.AveragePooling2D(pool_size=(3,3), strides=(2,2)))

alex_net.add(tf.keras.layers.Conv2D(384, 3, padding='same'))
alex_net.add(tf.keras.layers.Conv2D(384, 3, padding='same'))
alex_net.add(tf.keras.layers.Conv2D(256, 3, padding='same'))
alex_net.add(tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(2,2)))
alex_net.add(tf.keras.layers.Flatten())

alex_net.add(tf.keras.layers.Dense(4096))
alex_net.add(II.ActivationLinearizer(initial_eq='relu'))#tf.keras.layers.Activation('relu'))
alex_net.add(tf.keras.layers.Dropout(DROPOUT_RATE))

alex_net.add(tf.keras.layers.Dense(4096))
alex_net.add(II.ActivationLinearizer(initial_eq='relu'))#tf.keras.layers.Activation('relu'))
alex_net.add(tf.keras.layers.Dropout(DROPOUT_RATE))

alex_net.add(tf.keras.layers.Dense(10))
alex_net.add(tf.keras.layers.Activation('softmax'))

#alex_net.summary(line_length=100)

alex_net.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=[tf.keras.metrics.TopKCategoricalAccuracy()])
alex_net.fit(x=train_data, epochs=5, validation_data=val_data)

print("\n")

then = time.time()
alex_net.evaluate(val_data)
print("Took: ", str(int((time.time() - then) * 1000)) + "ms")

#Basic AlexNet (T5 Acc) -> 3.306s for 95-100% final validation accuracy (Training is ~50-60% top-5 acc)

#---------------------------------------------------------------------------------Piecewise Linear Units
#AlexNet w/1PLU (T5 Acc) -> 3.739s for 55% f.v.a. (Training is ~40-60%)

#AlexNet w/2PLU (T5 Acc) -> 5.439s for 100% f.v.a.  (Immediately got to 100% top-5 validation accuracy) (Training is ~50-100% depending on init)
#           NOTE: Sometimes loss gets EXORBITANTLY massive at least in training, validation remains around 2.3 loss
#``          `` (T4 Acc) -> 5.428s for 100% f.v.
#           NOTE: Same thing here, training loss gets exorbitantly high (742183.3125), but validation loss is always 2.3
#``          `` (T3 Acc) -> 5.438s for 100% f.v.
#``          `` (T2 Acc) -> 5.448s for 0% f.v. (Training is ~0-40% depending on init)

#AlexNet w/3PLU (T5 Acc) -> 5.494s for 100% f.v. (Training is ~80-100%)

#AlexNet w/4PLU (T5 Acc) -> 5.563s for 100% f.v. & 4082 loss (Training is ~50-98%)

#---------------------------------------------------------------------------------Activation Linearizer
#AlexNet w/1AL Init=relu (T5 Acc) -> 3.310s for 98-99% f.v. & 2.3 loss (Training is ~52-90-76%)

#AlexNet w/2AL Init=relu (T5 Acc) -> 3.317s for 100% f.v. & 2.3 loss (Training is ~50-95-80%)

#AlexNet w/3AL Init=relu (T5 Acc) -> 3.340s for 66% f.v. & 2.1 loss (max of 72% v.a.) (Training is ~55-60%)

#AlexNet w/4AL Init=relu (T5 Acc) -> 3.337s for 0% f.v. & 23.9 loss (max of 50% v.a.) (Training is ~55-60%)
