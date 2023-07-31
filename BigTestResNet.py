import tensorflow as tf
import tensorflow_datasets as tfds
import InferredActivations.Inferrer as II
import time
from keras import layers
from keras.metrics import TopKCategoricalAccuracy

IMAGE_SIZE = (224, 224, 3)
BATCH_SIZE = 16
AUGMENT_DATA = False
CONCATENATE_AUGMENTATION = False

print("Starting ResNet sandbox")
print("\t{} Batch Size\n\t({}, {}, {}) Image Size".format(BATCH_SIZE, IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE[2]))

print("\nLoading imagenet2012 training set")
train_ds, val_ds = tfds.load("imagenet2012", split=["train", "validation"], as_supervised=True, batch_size=BATCH_SIZE, download=False)
print("\t" + str(train_ds.cardinality() * BATCH_SIZE) + " training images loaded.")
print("\t" + str(val_ds.cardinality() * BATCH_SIZE) + " validation images loaded.")

"""
Olga Russakovsky*, Jia Deng*, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, 
Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg and Li Fei-Fei. (* = equal contribution) 
ImageNet Large Scale Visual Recognition Challenge. arXiv:1409.0575, 2014.
"""

resize_and_rescale = tf.keras.models.Sequential([
    layers.Resizing(IMAGE_SIZE[0], IMAGE_SIZE[1]),
    layers.Rescaling(1./255),
])

def preprocess1(x, y):
    return resize_and_rescale(x, training=True), y

train_data = train_ds.map(preprocess1)
val_data = val_ds.map(preprocess1)

if AUGMENT_DATA:
    print("\nAugmenting Data")
    augmentation = tf.keras.models.Sequential([
        layers.RandomZoom((0, -0.5)),
        layers.RandomFlip(),
        resize_and_rescale
    ])

    def preprocess2(x, y):
        return augmentation(x, training=True), y
    
    if CONCATENATE_AUGMENTATION:
        print("\tConcatenating Second Preprocess")
        train_data = train_data.concatenate(train_ds.map(preprocess2))
        print("\tIncreased Training Images to {}".format(train_data.cardinality() * BATCH_SIZE))
    else:
        train_data = train_ds.map(preprocess2)

print("\nDefining Resnet Layer")
#Copied from DifferentModels/ResNet.py file
class ResidualBlock(layers.Layer):
    def __init__(self, 
                 filters = (64, 64, 256),
                 stride=1,
                 activation_1 = layers.Activation("relu"),
                 activation_2 = layers.Activation("relu"),
                 activation_3 = layers.Activation("relu")):
        super(ResidualBlock, self).__init__()
        self.filters = filters
        self.stride = stride
        self.a1 = activation_1
        self.a2 = activation_2
        self.a3 = activation_3

    def build(self, input_shape):
        f1, f2, f3 = self.filters

        self.plain = tf.keras.models.Sequential([
            layers.Conv2D(f1, 1, strides=self.stride),
            layers.BatchNormalization(),
            self.a1,

            layers.Conv2D(f2, 3, padding='same'),
            layers.BatchNormalization(),
            self.a2,

            layers.Conv2D(f3, 1),
            layers.BatchNormalization(),
        ])

        is_expanding = not (input_shape[-1] == f3)
        if is_expanding:
            self.id_pass = tf.keras.models.Sequential([
                layers.Conv2D(f3, 1, strides=self.stride),
                layers.BatchNormalization(),
            ])
        else:
            self.id_pass = layers.BatchNormalization()

        self.final_pass = self.a3
    
    def call(self, input):
        return self.final_pass(self.plain(input) + self.id_pass(input))

print("\nEstablishing Multi-GPU Training Target")
strat = tf.distribute.MirroredStrategy()
print("\tDevices available: {}".format(strat.num_replicas_in_sync))

with strat.scope():
    res_net = tf.keras.models.Sequential([
        layers.Conv2D(64, 7, strides=2, input_shape=IMAGE_SIZE),
        layers.Activation('relu'),
        layers.MaxPooling(3, 2),

        ResidualBlock((64, 64, 256)),
        ResidualBlock((64, 64, 256)),
        ResidualBlock((64, 64, 256)),

        ResidualBlock((128, 128, 512), stride=2),
        ResidualBlock((128, 128, 512)),
        ResidualBlock((128, 128, 512)),

        ResidualBlock((256, 256, 1024), stride=2),
        ResidualBlock((256, 256, 1024)),
        ResidualBlock((256, 256, 1024)),
        ResidualBlock((256, 256, 1024)),
        ResidualBlock((256, 256, 1024)),
        ResidualBlock((256, 256, 1024)),

        ResidualBlock((512, 512, 2048), stride=2),
        ResidualBlock((512, 512, 2048)),
        ResidualBlock((512, 512, 2048)),

        layers.AveragePooling2D(2, padding='same'),
        layers.Flatten(),
        
        layers.Dense(1000),
        layers.Activation('softmax'),
    ])

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_T5', patience=20, mode='max', start_from_epoch=5, verbose=1)

    checks = [early_stop]

    metrics_to_use = [TopKCategoricalAccuracy(name="T5"), TopKCategoricalAccuracy(k=3, name="T3"), TopKCategoricalAccuracy(k=1,name="T1")]

res_net.summary(line_length=80)

print("\nTraining Residual Net:")
res_net.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalAccuracy, metrics=metrics_to_use)
res_net.fit(train_data, epochs=200, validation_data=val_data, callbacks=checks)

print("\n")
then = time.time()
res_net.evaluate(val_data)
print("Took {}ms".format(int( (time.time() - then) * 1000 )))