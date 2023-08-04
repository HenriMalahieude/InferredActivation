import numpy as np
import tensorflow as tf
import InferredActivations.Inferrer as II
import time
from MPTimeSeriesGenerator import MPTimeseriesGenerator
from keras import models, layers

#2023 Rendition of https://github.com/zpzim/LAMP-ICDM2019

BATCH_SIZE = 64
AMOUNT_OF_INPUT_STREAMS = 1
SEQ_LEN_T = BATCH_SIZE * 10
SEQ_LEN_V = BATCH_SIZE * 5

WINDOW_SIZE = 32
LOOK_AHEAD = 1
LOOK_BEHIND = 5

OUTPUT_SIZE = 256
INPUT_SIZE = (1, 1, OUTPUT_SIZE + (LOOK_BEHIND + LOOK_AHEAD))

print('Starting LAMP Sandbox with stats:')
print("\tBatch Size: {}\n\tMP_Window Size: {}\n\t\tLook Ahead: {}\n\t\tLook Behind: {}".format(BATCH_SIZE, WINDOW_SIZE, LOOK_AHEAD, LOOK_BEHIND))
print("\tInput Size: ({}, {}, {})\n\tOutput Size: {}".format(INPUT_SIZE[0], INPUT_SIZE[1], INPUT_SIZE[2], OUTPUT_SIZE))

print("\nCreating Train/Validation Data")
#Unprocessed Data
up_train_x = np.random.random(size=(SEQ_LEN_T, AMOUNT_OF_INPUT_STREAMS))
up_train_y = np.random.random(size=(SEQ_LEN_T - WINDOW_SIZE + 1, AMOUNT_OF_INPUT_STREAMS))

up_val_x = np.random.random(size=(SEQ_LEN_V, AMOUNT_OF_INPUT_STREAMS))
up_val_y = np.random.random(size=(SEQ_LEN_V - WINDOW_SIZE + 1, AMOUNT_OF_INPUT_STREAMS))

#Processed Generators
train_data = MPTimeseriesGenerator(up_train_x, up_train_y, 1, 
                                   batch_size=BATCH_SIZE, mp_window=WINDOW_SIZE, num_outputs=OUTPUT_SIZE, lookahead=(LOOK_AHEAD + OUTPUT_SIZE), lookbehind=LOOK_BEHIND, num_input_timeseries=AMOUNT_OF_INPUT_STREAMS)
val_data = MPTimeseriesGenerator(up_val_x, up_val_y, 1, 
                                 batch_size=BATCH_SIZE, mp_window=WINDOW_SIZE, num_outputs=OUTPUT_SIZE, lookahead=(LOOK_AHEAD + OUTPUT_SIZE), lookbehind=LOOK_BEHIND, num_input_timeseries=AMOUNT_OF_INPUT_STREAMS)

for x, y in train_data:
    print("X: ", x.shape, "Y: ", y.shape)
    break

print("\nInitializing Building Block")
class LAMP_ResNetBlock(layers.Layer):
    def __init__(self, n_feature_maps):
        super(LAMP_ResNetBlock, self).__init__()
        self.feature_maps = n_feature_maps

    def build(self, input_shape):
        self.conv_pass = models.Sequential([
            layers.Conv2D(self.feature_maps, (8,1), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),

            layers.Conv2D(self.feature_maps, (5,1), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),

            layers.Conv2D(self.feature_maps, (3,1), padding='same'),
            layers.BatchNormalization(),
        ])

        is_expanding = not (input_shape[-1] == self.feature_maps)
        if is_expanding:
            self.shortcut_pass = models.Sequential([
                layers.Conv2D(self.feature_maps, (1,1), padding='same'),
                layers.BatchNormalization(),
            ])
        else:
            self.shortcut_pass = layers.BatchNormalization()

        self.final_pass = layers.Activation('relu')
        
    
    def call(self, input):
        convolution = self.conv_pass(input)
        bypass = self.shortcut_pass(input)
        return self.final_pass(convolution + bypass)


print('\nInitializing Multi-GPU Workload')
strat = tf.distribute.MirroredStrategy()
print('\tUsing {} devices'.format(strat.num_replicas_in_sync))

with strat.scope():
    LAMP_Model = models.Sequential([
        layers.BatchNormalization(input_shape=INPUT_SIZE),

        LAMP_ResNetBlock(96),
        LAMP_ResNetBlock(192),
        LAMP_ResNetBlock(192),

        layers.Dense(OUTPUT_SIZE),
        layers.Activation('sigmoid'),
        layers.Reshape((OUTPUT_SIZE, 1)),
    ])

    #metrics_to_use = ['accuracy']

#LAMP_Model.summary()

#"""
print("\nTraining model")
LAMP_Model.compile(optimizer='adam', loss='mse')
LAMP_Model.fit(train_data, epochs=5, validation_data=val_data)

print("\n")

then = time.time()
LAMP_Model.evaluate(val_data)
print("Took: {}ms".format(int((time.time() - then) * 1000)))
#"""