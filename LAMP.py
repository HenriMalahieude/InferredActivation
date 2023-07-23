import tensorflow as tf
import InferredActivations.Inferrer as II
import time
from keras import models, layers

BATCH_SIZE = 128

INPUT_SIZE = 1#?
FEATURE_COUNT = 96
OUTPUT_SIZE = 256

print('Starting LAMP Sandbox')
print("\nLoading In Data?")


print("\nInitializing Building Blocks?")

print('\nInitializing Multi-GPU Workload')
strat = tf.distribute.MirroredStrategy()
print('\tUsing {} devices to train {} batches'.format(strat.num_replicas_in_sync, BATCH_SIZE))

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

with strat.scope():
    LAMP_Model = models.Sequential([
        layers.BatchNormalization(input=INPUT_SIZE),

        LAMP_ResNetBlock(FEATURE_COUNT),
        LAMP_ResNetBlock(FEATURE_COUNT*2),
        LAMP_ResNetBlock(FEATURE_COUNT*2),

        layers.Flatten(),
        layers.Dense(OUTPUT_SIZE),
        layers.Activation('sigmoid')
    ])

    metrics_to_use = ['accuracy']

LAMP_Model.summary()