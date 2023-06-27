import time
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import keras

from keras import losses, datasets

(x_train,y_train),(x_test,y_test) = datasets.mnist.load_data()

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

def quantizeBeforeTraining(mod):
    return tfmot.quantization.keras.quantize_model(mod)

def quantizeAfterTraining(mod, weights_only=False):
    converter = tf.lite.TFLiteConverter.from_keras_model(mod)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if (weights_only == False):
        def representative_data_gen():
            for x in range(200, 400):
                yield [x_train[x:x+1]]
        
        converter.representative_dataset = representative_data_gen

    new_model = tf.lite.Interpreter(model_content=converter.convert())
    new_model.allocate_tensors()

    return new_model

#Returns TrainTime, History
def trainAndTime(mod, ee=5, eager=False):
    start_t = time.time()
    mod.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'], run_eagerly=eager)
    hist = mod.fit(x_train, y_train, batch_size=64, epochs=ee, validation_data=(x_val, y_val))
    return time.time() - start_t, hist

def printGradients(mod, *nums):
    with tf.GradientTape() as tape:
        y = mod(x_train[:0])
        loss = losses.sparse_categorical_crossentropy(y_true = y_train[:0], y_pred = y)
        gradients = tape.gradient(loss, mod.trainable_weights)
        if (len(nums) <= 0):
            print(gradients)
        else:
            for i in range(len(nums)):
                print(gradients[nums[i]])
    print("\n")

#Tests a "Quantized" model on the data, returns success float, and time total float (ms)
def testQuantizedModel(mod, name):
    t_s = time.time()
    input_details = mod.get_input_details()
    output_details = mod.get_output_details()

    total = 0
    for x in range(len(x_val)):
        mod.set_tensor(input_details[0]['index'], x_val[x:x+1])
        mod.invoke()
        if tf.reduce_any(tf.math.not_equal(y_val[x:x+1], mod.get_tensor(output_details[0]['index']))):
            total = total + 1
    
    t_tot = (time.time() - t_s) * 1000
    print(name + " - total time: " + str(int(t_tot)) +"ms - val_accuracy: " + str(total/len(y_val)))
    return (total/len(y_val)), t_tot

#Takes base layer and maps trained weights to cloned, layerMapping is an array of pairs where (clone index, base index) maps to "equivalent" layers
def extractFromAndLockInto(base, clone, layerMapping):
    for i in range(len(layerMapping)):
        c_lay = layerMapping[i][0]
        b_lay = layerMapping[i][1]
        clone.layers[c_lay].set_weights(base.layers[b_lay].get_weights())
        clone.layers[c_lay].trainable = False

    return