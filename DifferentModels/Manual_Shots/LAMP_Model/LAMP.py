import numpy as np
import tensorflow as tf
import InferredActivations.Inferrer as II
from InferredActivations.Inferrer.ActivationLinearizer import OuterLockedBound, InnerLockedBound
import scipy.io as sio
from MPTimeSeriesGenerator import MPTimeseriesGenerator
from keras import models, layers

#2023 Rendition of https://github.com/zpzim/LAMP-ICDM2019

EPOCHS = 15
BATCH_SIZE = 1024
AMOUNT_OF_INPUT_STREAMS = 1

WINDOW_SIZE = 16
LOOK_AHEAD = 1
LOOK_BEHIND = 5

OUTPUT_SIZE = 256
INPUT_SIZE = (WINDOW_SIZE, AMOUNT_OF_INPUT_STREAMS, OUTPUT_SIZE + (LOOK_BEHIND + LOOK_AHEAD))
DATASET_PATH = "LAMP_Datasets/LCCB_dataset.mat"

MODE = "Control"
assert MODE in ["Control", "Control(Tanh)", "PWLU", "AL(Sig)", "AL(Tanh)"] #TODO: PWLU and AL(Sig) and Control(Tanh)

def ultra_fast_sigmoid(x): #Stolen directly from theano github, converted to tensor functions
    x = 0.5 * x
    # The if is a tanh approximate.
    b1 = OuterLockedBound(x, -3, top=False)
    b2 = InnerLockedBound(x, -3, -1.7)
    b3 = InnerLockedBound(x, -1.7, 0)
    b4 = InnerLockedBound(x, 0, 1.7)
    b5 = InnerLockedBound(x, 1.7, 3)
    b6 = OuterLockedBound(x, 3)

    x1 = b1 * -0.99505475368673
    x2 = b2 * (-0.935409070603099 + 0.0458812946797165 * (x + 1.7))
    x3 = b3 * tf.math.divide_no_nan((1.5 * x), (1 - x))
    x4 = b4 * tf.math.divide_no_nan((1.5 * x), (1 + x))
    x5 = b5 * (0.935409070603099 + 0.0458812946797165 * (x - 1.7))
    x6 = b6 * 0.99505475368673

    z = x1 + x2 + x3 + x4 + x5 + x6 #each element should be mutually exclusive so only one operation per element technically

    return 0.5 * (z + 1.0) 

final_control_act = II.fastexp512_act()

print('Starting LAMP Sandbox with stats:')
print("\tBatch Size: {}\n\tMP_Window Size: {}\n\t\tLook Ahead: {}\n\t\tLook Behind: {}".format(BATCH_SIZE, WINDOW_SIZE, LOOK_AHEAD, LOOK_BEHIND))
print("\tInput Size: ({}, {}, {})\n\tOutput Size: {}".format(INPUT_SIZE[0], INPUT_SIZE[1], INPUT_SIZE[2], OUTPUT_SIZE))

print("\nLoading Train/Validation Data")
#Unprocessed Data
all_data = sio.loadmat(DATASET_PATH)

mp = np.array(all_data["mp_train"])
ts = np.array(all_data["ts_train"])

mp_val = np.array(all_data["mp_val"])
ts_val = np.array(all_data["ts_val"])

#Processed Generators
train_data = MPTimeseriesGenerator(ts, mp, WINDOW_SIZE,
                                   batch_size=BATCH_SIZE, mp_window=WINDOW_SIZE, num_outputs=OUTPUT_SIZE, lookahead=(LOOK_AHEAD + OUTPUT_SIZE), lookbehind=LOOK_BEHIND, num_input_timeseries=AMOUNT_OF_INPUT_STREAMS)
val_data = MPTimeseriesGenerator(ts_val, mp_val, WINDOW_SIZE,
                                 batch_size=BATCH_SIZE, mp_window=WINDOW_SIZE, num_outputs=OUTPUT_SIZE, lookahead=(LOOK_AHEAD + OUTPUT_SIZE), lookbehind=LOOK_BEHIND, num_input_timeseries=AMOUNT_OF_INPUT_STREAMS)

print("\nInitializing Building Block")
class LAMP_ResNetBlock(layers.Layer):
    def __init__(self, n_feature_maps):
        super(LAMP_ResNetBlock, self).__init__()
        self.feature_maps = n_feature_maps

    def build(self, input_shape):
        self.conv_pass = models.Sequential([
            layers.Conv2D(self.feature_maps, (8,1), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu') if MODE.startswith("Control") else (II.ActivationLinearizer() if MODE.startswith("AL") else II.PiecewiseLinearUnitV1()),

            layers.Conv2D(self.feature_maps, (5,1), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu') if MODE.startswith("Control") else (II.ActivationLinearizer() if MODE.startswith("AL") else II.PiecewiseLinearUnitV1()),

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

        self.final_pass = layers.Activation('relu') if MODE.startswith("Control") else (II.ActivationLinearizer() if MODE.startswith("AL") else II.PiecewiseLinearUnitV1())
        
    
    def call(self, input):
        convolution = self.conv_pass(input)
        bypass = self.shortcut_pass(input)
        return self.final_pass(convolution + bypass)

print('\nInitializing Multi-GPU Workload')
strat = tf.distribute.MirroredStrategy()
print('\tUsing {} devices'.format(strat.num_replicas_in_sync))

with strat.scope():
    f_actC = final_control_act if MODE == "Control" else layers.Activation("tanh")
    f_actA = II.ActivationLinearizer("sigmoid") if MODE == "AL(Sig)" else II.ActivationLinearizer("tanh")
    LAMP_Model = models.Sequential([
        layers.BatchNormalization(input_shape=INPUT_SIZE),

        LAMP_ResNetBlock(96),
        LAMP_ResNetBlock(192),
        LAMP_ResNetBlock(192), #Trim this off if still NaN

        layers.Flatten(),
        layers.Dense(OUTPUT_SIZE),
        f_actC if MODE.startswith("Control") else (f_actA if MODE.startswith("AL") else II.PiecewiseLinearUnitV1()), #layers.Activation("tanh") as recommended
        layers.Reshape((OUTPUT_SIZE, 1)),
    ])

    #metrics_to_use = ['accuracy']

LAMP_Model.summary()

if MODE.startswith("AL"):
    print("{} init".format(LAMP_Model.layers[6].initialization))

#"""
print("\nTraining model {}".format(MODE))
LAMP_Model.compile(optimizer='adam', loss='mse')
hist = LAMP_Model.fit(train_data, epochs=EPOCHS, validation_data=val_data, callbacks=[tf.keras.callbacks.TerminateOnNaN()])

print(hist.history["loss"])
print(hist.history["val_loss"])

#"""

#3 Epochs, single top change
#Control Sigmoid       T-Loss: 0.0026, V-Loss: 0.0040
#Control Tanh          T-Loss: 0.0026, V-Loss: 0.0050

#Piecewise Linear Unit T-Loss: 0.0025, V-Loss: 0.0041
#Activation Linearizer T-Loss: 0.2038, V-Loss: 0.1810

#Fully PWLU             T-Loss: 0.0025, V-Loss: 0.0049
#Fully AL              T-Loss: 0.2235, V-Loss: 0.4191 (Best was 0.1225, 0.1179)