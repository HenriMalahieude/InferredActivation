import tensorflow as tf
import InferredActivations.Inferrer as II
import time
from keras import layers, models

print("Starting LesNet Sandbox")
print("\nLoading Training Data from MNIST")
(x_train,y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = tf.pad(x_train, [[0, 0], [2,2], [2,2]])/255
x_test = tf.pad(x_test, [[0, 0], [2,2], [2,2]])/255

#LesNet requires 32x32
x_train = tf.expand_dims(x_train, axis=3, name=None)
x_test = tf.expand_dims(x_test, axis=3, name=None)

#Remove some data so we can validate
x_val = x_train[-2000:,:,:,:] 
y_val = y_train[-2000:] 
x_train = x_train[:-2000,:,:,:] 
y_train = y_train[:-2000]

print("\nEstablish Multi-GPU Training Target")
strat = tf.distribute.MirroredStrategy()
print("\tAvailable Devices: {}".format(strat.num_replicas_in_sync))
print("\tBuilding Model")
with strat.scope():
    lesnet = models.Sequential([
        layers.Conv2D(6, 5, input_shape=x_train.shape[1:]),
        layers.Activation('tanh'),
        layers.AveragePooling2D(2),
        layers.Activation('sigmoid'),

        layers.Conv2D(16, 5),
        layers.Activation('tanh'),
        layers.AveragePooling2D(2),
        layers.Activation('sigmoid'),

        layers.Conv2D(120, 5),
        layers.Activation('tanh'),
        layers.Flatten(),

        layers.Dense(84),
        layers.Activation('tanh'),
        layers.Dense(10),
        layers.Activation('softmax')
    ])

print("\nTraining Model")
lesnet.compile(optimizer='adam', loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
hist = lesnet.fit(x_train, y_train, batch_size=64, epochs=5, validation_data=(x_val, y_val))

print("\n")
then = time.time()
lesnet.evaluate((x_test, y_test))
print("Took {}ms".format(int((time.time() - then) * 1000)))