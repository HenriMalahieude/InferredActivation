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
        II.LinearMaxV4()#layers.Activation('softmax') #
    ])

print("\nTraining Model")
lesnet.compile(optimizer='adam', loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
hist = lesnet.fit(x_train, y_train, batch_size=64, epochs=50, validation_data=(x_val, y_val))

print("\n")
then = time.time()
lesnet.evaluate(x=x_test, y=y_test)
print("Took {}ms".format(int((time.time() - then) * 1000)))


#------------------------------------------------------------------Mini Epoch Area (5)
#Control ------------> 0.661s for 94% Test Acc. (96% FVA, 93% FTA)

#6/7Relus -----------> 0.700s for 98% Test Acc. (99% FVA, 98% FTA)

#6/7ALs w/== Init ---> 1.045s for 11% Test Acc. (11% FVA, 11% FTA)
#6/7ALs w/Relu Init -> 1.036s for 98% Test Acc. (99% FVA, 99% FTA)

#6/7PLUs ------------> 1.325s for 98% Test Acc. (98% FVA, 98% FTA)

#1/7LinearMaxV1 -----> 0.734s for 77% Test Acc. (80% FVA, 58% FTA)   NOTE: Best Epoch-3 91% FVA, 82% FTA
#1/7LinearMaxV1.5 ---> 0.759s for 10% Test Acc. (10% FVA, 37% FTA)   NOTE: Best Epoch-4 93% FVA, 82% FTA

#1/7LinearMaxV2 -----> 0.671s for 9.7% Test Acc (9.7% FVA, 9.9% FTA) NOTE: Best Epoch-2 17% FVA, 11% FTA

#1/7LinearMaxV3 -----> 0.667s for 95% Test Acc. (97% FVA, 95% FTA)

#1/7LinearMaxV4 -----> 0.723s for 42% Test Acc. (42% FVA, 26% FTA)   NOTE: Best Epoch-4 85% FVA, 73% FTA
#1/7LinearMaxV4.5 ---> 0.714s for 70% Test Acc. (74% FVA, 61% FTA)



#------------------------------------------------------------------Mega Epoch Area (50)
#Control ------------> 0.736s for 98.14% Test Acc. (98.75% FVA, 98.22% FTA) NOTE: Best Epoch-33 99.00% FVA, 97.74% FTA

#6/7Relus -----------> 0.736s for 99.07% Test Acc. (99.25% FVA, 99.96% FTA) NOTE: Best Epoch-29 99.55% FVA, 99.92% FTA
#6/7PLUs ------------> 1.467s for 98.36% Test Acc. (99.10% FVA, 99.94% FTA) NOTE: Best Epoch-49 99.35% FVA, 99.92% FTA
#6/7ALs w/== Init ---> 1.185s for 11.35% Test Acc. (10.70% FVA, 11.26% FTA)
#6/7ALs w/Relu Init -> 1.191s for 98.53% Test Acc. (99.05% FVA, 99.31% FTA) NOTE: Best Epoch-42 66.60% FVA, 99.90% FTA

#1/7LinearMaxV1 -----> 0.765s for 89.69% Test Acc. (93.15% FVA, 83.89% FTA) NOTE: Best Epoch-24 95.75% FVA, 89.02% FTA ; Worst Epoch-39 09.95% FVA, 09.81%
#1/7LinearMaxV1.5 ---> 0.774s for 09.80% Test Acc.                          NOTE: Best Epoch-23 96.00% FVA, 89.49% FTA
#1/7LinearMaxV2 -----> quickly NaN, may be division by zero
#1/7LinearMaxV3 -----> 0.752s for 98.61% Test Acc. (99.15% FVA, 99.13% FTA)
#1/7LinearMaxV4 -----> 0.711s for 88.72% Test Acc. (90.90% FVA, 84.55% FTA) NOTE: Best Epoch-49 92.40% FVA, 85.29% FTA
#1/7LinearMaxV4.5 ---> quickly NaN may be division by zero