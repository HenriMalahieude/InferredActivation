import tensorflow as tf
import tensorflow_datasets as tfds
import InferredActivations.Inferrer as II
import time
from keras import layers
from keras.metrics import TopKCategoricalAccuracy

IMAGE_SIZE = (224, 224, 3)
BATCH_SIZE = 64
DROPOUT_RATE = 0.5
AUGMENT_DATA = True

print("Starting ResNet sandbox")
print("\nLoading imagenette/320px-v2 into {} batch size".format(BATCH_SIZE))
train_ds, val_ds = tfds.load("imagenette/320px-v2", split=['train', 'validation'], as_supervised=True, batch_size=BATCH_SIZE)

prepare_img = tf.keras.models.Sequential([
    layers.Resizing(IMAGE_SIZE[0], IMAGE_SIZE[1]),
    layers.Rescaling(1./255),
])

def preprocess1(x, y):
    return prepare_img(x, training=True), y

print("\tPreparing Data")
train_data = train_ds.map(preprocess1)
val_data = val_ds.map(preprocess1)

print("\tTraining Set is: {}".format(train_data.cardinality() * BATCH_SIZE))
for train_x, train_y in train_data:
    print("\tTrain Shape is Input:", train_x[0].shape, "Label:", train_y[0].shape)
    #print(train_y)
    break

if AUGMENT_DATA:
    print("\nAugmenting Data")

    augmenter = tf.keras.models.Sequential([
        layers.Resizing(IMAGE_SIZE[0]*2, IMAGE_SIZE[1]*2),
        layers.RandomCrop(IMAGE_SIZE[0], IMAGE_SIZE[0]),
        layers.RandomFlip(),
        layers.Rescaling(1./255)
    ])
    def preprocess2(x, y):
        return augmenter(x, training=True), y
    
    train_data = train_data.concatenate(train_data.map(preprocess2))
    print("\tTraining Set now: {}".format(train_data.cardinality() * BATCH_SIZE))

print("\nDefining Resnet Layer")
#Some of this shit doesn't feel right, but I can't decipher the ResNet paper that well for some reason
#Using this as reference: https://machinelearningknowledge.ai/keras-implementation-of-resnet-50-architecture-from-scratch/
class ResidualBlock(layers.Layer):
    def __init__(self, 
                 filters = (64, 64, 256), 
                 dimension_increase=False, 
                 stride=1,
                 activation_1 = layers.Activation("relu"),
                 activation_2 = layers.Activation("relu"),
                 activation_3 = layers.Activation("relu")):
        super(ResidualBlock, self).__init__()
        self.filters = filters
        self.stride = stride
        self.dimension_increase = dimension_increase
        self.a1 = activation_1
        self.a2 = activation_2
        self.a3 = activation_3

    def build(self, input_shape):
        f1, f2, f3 = self.filters

        self.plain = tf.keras.models.Sequential([
            layers.Conv2D(f1, 1, strides=self.stride),
            #layers.BatchNormalization(),
            self.a1,

            layers.Conv2D(f2, 3, padding='same'),
            #layers.BatchNormalization(),
            self.a2,

            layers.Conv2D(f3, 1),
            #layers.BatchNormalization(),
        ])

        if self.dimension_increase:
            self.id_pass = tf.keras.models.Sequential([
                layers.Conv2D(f3, 1, strides=self.stride),
                #layers.BatchNormalization(),
            ])

        self.final_pass = self.a3
    
    def call(self, input):
        y = self.plain(input)
        
        if self.dimension_increase:
            input = self.id_pass(input)

        y = self.final_pass(y + input)
        return y


print("\nInitializing Model w/Multi-GPU Training")
strat = tf.distribute.MirroredStrategy()
print("\tDevices to be trained on: {}".format(strat.num_replicas_in_sync))
print("\tInstantiating Model")
with strat.scope():
    res_net = tf.keras.models.Sequential([
        #layers.ZeroPadding2D((3, 3), input_shape=IMAGE_SIZE),
        layers.Conv2D(64, 7, strides=2, input_shape=IMAGE_SIZE),
        #layers.BatchNormalization(),
        II.ActivationLinearizer(initial_eq='relu'),#layers.Activation('relu'),
        layers.MaxPooling2D(3, 2),

        ResidualBlock((64, 64, 256), dimension_increase=True, activation_1=II.ActivationLinearizer(initial_eq='relu'), activation_2=II.ActivationLinearizer(initial_eq='relu'), activation_3=II.ActivationLinearizer(initial_eq='relu')),
        ResidualBlock((64, 64, 256), activation_1=II.ActivationLinearizer(initial_eq='relu'), activation_2=II.ActivationLinearizer(initial_eq='relu'), activation_3=II.ActivationLinearizer(initial_eq='relu')),
        ResidualBlock((64, 64, 256), activation_1=II.ActivationLinearizer(initial_eq='relu'), activation_2=II.ActivationLinearizer(initial_eq='relu'), activation_3=II.ActivationLinearizer(initial_eq='relu')),

        ResidualBlock((128, 128, 512), dimension_increase=True, stride=2, activation_1=II.ActivationLinearizer(initial_eq='relu'), activation_2=II.ActivationLinearizer(initial_eq='relu'), activation_3=II.ActivationLinearizer(initial_eq='relu')),
        ResidualBlock((128, 128, 512), activation_1=II.ActivationLinearizer(initial_eq='relu'), activation_2=II.ActivationLinearizer(initial_eq='relu'), activation_3=II.ActivationLinearizer(initial_eq='relu')),
        ResidualBlock((128, 128, 512), activation_1=II.ActivationLinearizer(initial_eq='relu'), activation_2=II.ActivationLinearizer(initial_eq='relu'), activation_3=II.ActivationLinearizer(initial_eq='relu')),

        ResidualBlock((256, 256, 1024), dimension_increase=True, stride=2, activation_1=II.ActivationLinearizer(initial_eq='relu'), activation_2=II.ActivationLinearizer(initial_eq='relu'), activation_3=II.ActivationLinearizer(initial_eq='relu')),
        ResidualBlock((256, 256, 1024), activation_1=II.ActivationLinearizer(initial_eq='relu'), activation_2=II.ActivationLinearizer(initial_eq='relu'), activation_3=II.ActivationLinearizer(initial_eq='relu')),
        ResidualBlock((256, 256, 1024), activation_1=II.ActivationLinearizer(initial_eq='relu'), activation_2=II.ActivationLinearizer(initial_eq='relu'), activation_3=II.ActivationLinearizer(initial_eq='relu')),
        ResidualBlock((256, 256, 1024), activation_1=II.ActivationLinearizer(initial_eq='relu'), activation_2=II.ActivationLinearizer(initial_eq='relu'), activation_3=II.ActivationLinearizer(initial_eq='relu')),
        ResidualBlock((256, 256, 1024), activation_1=II.ActivationLinearizer(initial_eq='relu'), activation_2=II.ActivationLinearizer(initial_eq='relu'), activation_3=II.ActivationLinearizer(initial_eq='relu')),
        ResidualBlock((256, 256, 1024), activation_1=II.ActivationLinearizer(initial_eq='relu'), activation_2=II.ActivationLinearizer(initial_eq='relu'), activation_3=II.ActivationLinearizer(initial_eq='relu')),

        ResidualBlock((512, 512, 2048), dimension_increase=True, stride=2, activation_1=II.ActivationLinearizer(initial_eq='relu'), activation_2=II.ActivationLinearizer(initial_eq='relu'), activation_3=II.ActivationLinearizer(initial_eq='relu')),
        ResidualBlock((512, 512, 2048), activation_1=II.ActivationLinearizer(initial_eq='relu'), activation_2=II.ActivationLinearizer(initial_eq='relu'), activation_3=II.ActivationLinearizer(initial_eq='relu')),
        ResidualBlock((512, 512, 2048), activation_1=II.ActivationLinearizer(initial_eq='relu'), activation_2=II.ActivationLinearizer(initial_eq='relu'), activation_3=II.ActivationLinearizer(initial_eq='relu')),

        layers.AveragePooling2D(2, padding='same'),
        layers.Flatten(),
        
        layers.Dense(256),
        II.ActivationLinearizer(initial_eq='relu'),#layers.Activation('relu'), #NOTE: Somehow removing the activations improvements it?

        layers.Dense(128),
        II.ActivationLinearizer(initial_eq='relu'),#layers.Activation('relu'),

        #layers.Dense(64),
        #layers.Activation('relu'),

        layers.Dense(10),
        layers.Activation('softmax')
    ])

    metrics_to_use = [TopKCategoricalAccuracy(name='T5'), TopKCategoricalAccuracy(name='T3', k=3), TopKCategoricalAccuracy(name='T2', k=2)]

#res_net.summary()


print("\nTraining Model, Stats:")
print("\tDropout: {}".format(DROPOUT_RATE))

#optim = tf.keras.optimizers.Adam(learning_rate=0.0005) #NOTE: Seems that I can't do anything to the optim. or it will just freeze forever

res_net.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=metrics_to_use)
res_net.fit(train_data, epochs=5, validation_data=val_data)

print("\n")
then = time.time()
res_net.evaluate(val_data)
print("Took: {}ms".format(int((time.time() - then) * 1000)))
#"""

#Control* ->  5.313s for  93% FVT5A,  76% FVT3A,  42% FVT2A                       (Training is T5 ~66-36-69%, T3 ~36-14-43%, T2 ~23-06-21%)
#           NOTE: (w/o Last ReLUs & w/o 1 Dense Layer)

#Control* ->  5.243s for 100% FVT5A,   0% FVT32A,                                 (Training is T5 ~79-100%  , T3 ~49-88-87%, T2 ~34-79-63%)
#           NOTE: Dense 256 -> 128 -> 10 with ReLU seps & No Batch Normalization


#2 PLUs   ->  5.227s for 100% FVT5A,   0% FVT32A,                                 (Training is T5 ~60-50%   , T3 ~30-32%   , T2 ~19-28-27%)
#. . .
#11 PLUs  ->  6.267s for 100% FVT5A,   0% FVT32A,           but 100% T3 Epoch 2-4 (Training is T5 ~58-53%   , T3 ~28-42%   , T2 ~16-29%   )
#. . .
#29 PLUs  -> 14.653s for 100% FVT5A,   0% FVT32A,                                 (Training is T5 ~63-54-56%, T3 ~43-39-42%, T2 ~30-36-24%)
#. . .
#38 PLUs  -> 22.021s for 100% FVT5A,   0% FVT32A,           but 100% T3 Epoch 1-4 (Training is T5 ~64-69-65%, T3 ~41-37%   , T2 ~22-30%   )
#. . .
#48 PLUs  -> 43.678s for 100% FVT3A,   0% FVT2A ,         100% T3 since Epoch 2-5 (Training is T5 ~59-65-64%, T3 ~40-42-38%, T2 ~18-23%   )


#2 ALs ---->  5.224s for 100% FVT5A,   0% FVT32A,           but  20% T2 Epoch 1   (Training is T5 ~62-71-63%, T3 ~42-47-39%, T2 ~30-22%   )
#. . .
#11 ALs --->  5.374s for  97% FVT5A,  96% FVT3A,  96% FVT2A                       (Training is T5 ~48-63-58%, T3 ~27-39-37%, T2 ~17-27%   )
#. . .
#29 ALs --->  7.619s for 100% FVT5A,   2% FVT3A,   2% FVT2A                       (Training is T5 ~63-72-69%, T3 ~35-45-34%, T2 ~22-29-21%)
#. . .
#38 ALs --->  9.400s for   0% FVT532A,                      but 100% T2 Epoch 2&4 (Training is T5 ~72-57-67%, T3 ~55-36-48%, T2 ~34-24-35%)
#. . .
#48 ALs ---> 11.899s for  49% FVT5A,  26% FVT3A,  11% FVT2A but 93% T5 Epoch 2    (Training is T5 ~65-72-64%, T3 ~42-50-45%, T2 ~24-31%   )