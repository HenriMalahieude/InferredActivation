import tensorflow as tf
import numpy as np
import InferredActivations.Inferrer as II
import scipy.io as sio
from MPTimeSeriesGenerator import MPTimeseriesGenerator
from keras import models, layers

BATCH_SIZE = 256
AMOUNT_OF_INPUT_STREAMS = 1

WINDOW_SIZE = 16
LOOK_AHEAD = 2
LOOK_BEHIND = 4

OUTPUT_SIZE = 16
INPUT_SIZE = (WINDOW_SIZE, AMOUNT_OF_INPUT_STREAMS, OUTPUT_SIZE + (LOOK_BEHIND + LOOK_AHEAD))
DATASET_PATH = "LAMP_Datasets/LCCB_dataset.mat"

#Knowledge Distillation Constants
TEACHER_EPOCHS = 1
STUDENT_EPOCHS = 1 #Remember that this will be on multiple increments, so it will be a total of 4 epochs if set to 1 student epoch
STUDENT_DS_RATIO = 0.15 #ratio of time series data that student will train on

print('Starting Knowledge Distillation')
print("\tBatch Size: {}\n\tMP_Window Size: {}\n\t\tLook Ahead: {}\n\t\tLook Behind: {}".format(BATCH_SIZE, WINDOW_SIZE, LOOK_AHEAD, LOOK_BEHIND))
print("\tInput Size: ({}, {}, {})\n\tOutput Size: {}".format(INPUT_SIZE[0], INPUT_SIZE[1], INPUT_SIZE[2], OUTPUT_SIZE))

print("\nLoading Train/Validation Data")
#Unprocessed Data
all_data = sio.loadmat(DATASET_PATH)

percnt = int(len(all_data["mp_train"]) * STUDENT_DS_RATIO)
mp_partial = np.array(all_data["mp_train"][:percnt])
ts_partial = np.array(all_data["ts_train"][:percnt])

mp_full = np.array(all_data["mp_train"])
ts_full = np.array(all_data["ts_train"])

mp_val = np.array(all_data["mp_val"])
ts_val = np.array(all_data["ts_val"])

#Processed Generators
print("\tProcessing Dataset")
train_data_f = MPTimeseriesGenerator(ts_full, mp_full, WINDOW_SIZE, 
                                   batch_size=BATCH_SIZE, mp_window=WINDOW_SIZE, num_outputs=OUTPUT_SIZE, lookahead=(LOOK_AHEAD + OUTPUT_SIZE), lookbehind=LOOK_BEHIND, num_input_timeseries=AMOUNT_OF_INPUT_STREAMS)
train_data_p = MPTimeseriesGenerator(ts_partial, mp_partial, WINDOW_SIZE, 
                                   batch_size=BATCH_SIZE, mp_window=WINDOW_SIZE, num_outputs=OUTPUT_SIZE, lookahead=(LOOK_AHEAD + OUTPUT_SIZE), lookbehind=LOOK_BEHIND, num_input_timeseries=AMOUNT_OF_INPUT_STREAMS)
val_data = MPTimeseriesGenerator(ts_val, mp_val, WINDOW_SIZE, 
                                 batch_size=BATCH_SIZE, mp_window=WINDOW_SIZE, num_outputs=OUTPUT_SIZE, lookahead=(LOOK_AHEAD + OUTPUT_SIZE), lookbehind=LOOK_BEHIND, num_input_timeseries=AMOUNT_OF_INPUT_STREAMS)

print("\nInitializing Building Block")
class LAMP_ResNetBlock(layers.Layer):
    def __init__(self, n_feature_maps, a1=layers.Activation('relu'), a2=layers.Activation('relu'), a3=layers.Activation('relu')):
        super(LAMP_ResNetBlock, self).__init__()
        self.feature_maps = n_feature_maps
        self.activation1 = a1
        self.activation2 = a2
        self.activation3 = a3

    def build(self, input_shape):
        self.conv_pass = models.Sequential([
            layers.Conv2D(self.feature_maps, (8,1), padding='same'),
            layers.BatchNormalization(),
            self.activation1,

            layers.Conv2D(self.feature_maps, (5,1), padding='same'),
            layers.BatchNormalization(),
            self.activation2,

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
        
    def call(self, input):
        convolution = self.conv_pass(input)
        bypass = self.shortcut_pass(input)
        return self.activation3(convolution + bypass)

print('\nInitializing Multi-GPU Workload')
strat = tf.distribute.MirroredStrategy()
print('\tUsing {} devices'.format(strat.num_replicas_in_sync))

control_arch = [
    #0
    {
        "lyrFn": layers.BatchNormalization,
        "args": [],
        "kwargs": {"input_shape": INPUT_SIZE}
    },

    #1
    {
        "lyrFn": LAMP_ResNetBlock,
        "args": [96],
        "kwargs": {}
    },

    #2
    {
        "lyrFn": LAMP_ResNetBlock,
        "args": [192],
        "kwargs": {}
    },

    #3
    {
        "lyrFn": LAMP_ResNetBlock,
        "args": [192],
        "kwargs": {}
    },

    #4
    {
        "lyrFn": layers.Flatten,
        "args": [],
        "kwargs": {}
    },

    #5
    {
        "lyrFn": layers.Dense,
        "args": [OUTPUT_SIZE],
        "kwargs": {}
    },

    #6
    {
        "lyrFn": layers.Activation,
        "args": ['sigmoid'],
        "kwargs": {}
    },

    #7
    {
        "lyrFn": layers.Reshape,
        "args": [(OUTPUT_SIZE, 1)],
        "kwargs": {}
    },
]

build_up_levels = [1, 2, 3] #Indices at which Knowledge Distillation should collect data
collected_data = [] #this will be a collection of tensors which will hold the data for 

with strat.scope():
    cllbcks_to_use = [tf.keras.callbacks.TerminateOnNaN()]

    teacher = models.Sequential([
        layers.BatchNormalization(input_shape=INPUT_SIZE),

        LAMP_ResNetBlock(96),
        LAMP_ResNetBlock(192),
        LAMP_ResNetBlock(192),

        layers.Flatten(),
        layers.Dense(OUTPUT_SIZE),
        layers.Activation('sigmoid'),
        layers.Reshape((OUTPUT_SIZE, 1)),
    ])

    teacher.compile(optimizer='adam', loss='mse')

print("\nTraining Teacher")
teacher.fit(train_data_f, epochs=TEACHER_EPOCHS, validation_data=val_data, callbacks=cllbcks_to_use)

print("\nAcquiring new 'labels'")
build_up_levels.reverse()
for i in build_up_levels:
    collected_data.append([])
    print("\tFor Index {}".format(i))
    lvl_c = i + 1 #For level count, since the og is actually an index
    while len(teacher.layers) > lvl_c:
        teacher.pop()
    
    #Collect Training Data for the Student
    for x, y in train_data_p:
        results = teacher.call(tf.convert_to_tensor(x))
        collected_data[-1].append(results)
            

#NOTE: Remember that build_up_levels is now [3, 2, 1] and so is collected_data

print("\nStarting Distillation")
print(len(collected_data), collected_data[0][0].shape)

#This is getting annoying because of the memory requirement, moving on to AlexNet for KD instead