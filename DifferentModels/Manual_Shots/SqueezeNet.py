import tensorflow as tf
import tensorflow_datasets as tfds
import InferredActivations.Inferrer as II
import time
from keras import models, layers
from keras.metrics import TopKCategoricalAccuracy

BATCH_SIZE = 128
IMAGE_SIZE = (224, 224, 3)
AUGMENT_DATA = True

print('Starting SqueezeNet Sandbox')
print('\nLoading imagenette/320px-v2 dataset')
train_ds, val_ds = tfds.load("imagenette/320px-v2", split=["train", "validation"], as_supervised=True, batch_size=BATCH_SIZE)

resize_and_rescale = models.Sequential([
    layers.Resizing(IMAGE_SIZE[0], IMAGE_SIZE[1]),
    layers.Rescaling(1./255.)
])

def preprocess1(x, y):
    return resize_and_rescale(x, training=True), y

train_data = train_ds.map(preprocess1)
val_data = val_ds.map(preprocess1)

if AUGMENT_DATA:
    print("\nAugmenting Data")
    augmenter = models.Sequential([
        layers.Resizing(IMAGE_SIZE[0] * 2, IMAGE_SIZE[1] * 2),
        layers.RandomCrop(IMAGE_SIZE[0], IMAGE_SIZE[1]),
        layers.RandomFlip(),
        layers.Rescaling(1./255)
    ])

    def preprocess2(x, y):
        return augmenter(x, training=True), y
    
    train_data = train_data.concatenate(train_ds.map(preprocess2))
    print("\tTraining Data is now {}".format(train_data.cardinality() * BATCH_SIZE))


print("\nDefining the Fire Module")
class FireModule(layers.Layer):
    def __init__(self, squeeze=16, expand=64 , a1=layers.Activation('relu'), a2=layers.Activation('relu'), a3=layers.Activation('relu')):
        super(FireModule, self).__init__()
        self.squeeze=squeeze
        self.expand=expand

        self.activation1 = a1
        self.activation2 = a2
        self.activation3 = a3

    def build(self, input_shape):
        self.sLayer = models.Sequential([
            layers.Conv2D(self.squeeze, 1),
            self.activation1,
        ])

        self.eOneLayer = models.Sequential([
            layers.Conv2D(self.expand, 1),
            self.activation2,
        ])

        self.eThreeLayer = models.Sequential([
            layers.Conv2D(self.expand, 3, padding='same'),
            self.activation3,
        ])

        self.sLayer.build(input_shape)
        self.eOneLayer.build(self.sLayer.compute_output_shape(input_shape))
        self.eThreeLayer.build(self.sLayer.compute_output_shape(input_shape))

    def call(self, input):
        x = self.sLayer(input)

        left = self.eOneLayer(x)
        right = self.eThreeLayer(x)

        return layers.concatenate([left, right], axis=3)
    
print("\nEstablishing Multi-GPU Training Target")
strat = tf.distribute.MirroredStrategy()
print("\tDevices to train on: {}".format(strat.num_replicas_in_sync))

with strat.scope():
    squeeze_net = models.Sequential([ #Standard, no bypasses
        layers.Conv2D(96, 7, strides=2, input_shape=IMAGE_SIZE),
        layers.MaxPooling2D(3, 2),
        FireModule(a1=II.ActivationLinearizer(initial_eq='relu'), a2=II.ActivationLinearizer(initial_eq='relu'), a3=II.ActivationLinearizer(initial_eq='relu')),
        FireModule(a1=II.ActivationLinearizer(initial_eq='relu'), a2=II.ActivationLinearizer(initial_eq='relu'), a3=II.ActivationLinearizer(initial_eq='relu')),
        FireModule(squeeze=32, expand=128, a1=II.ActivationLinearizer(initial_eq='relu'), a2=II.ActivationLinearizer(initial_eq='relu'), a3=II.ActivationLinearizer(initial_eq='relu')),
        layers.MaxPooling2D(3, 2),
        FireModule(squeeze=32, expand=128, a1=II.ActivationLinearizer(initial_eq='relu'), a2=II.ActivationLinearizer(initial_eq='relu'), a3=II.ActivationLinearizer(initial_eq='relu')),
        FireModule(squeeze=48, expand=192, a1=II.ActivationLinearizer(initial_eq='relu'), a2=II.ActivationLinearizer(initial_eq='relu'), a3=II.ActivationLinearizer(initial_eq='relu')),
        FireModule(squeeze=48, expand=192, a1=II.ActivationLinearizer(initial_eq='relu'), a2=II.ActivationLinearizer(initial_eq='relu'), a3=II.ActivationLinearizer(initial_eq='relu')),
        FireModule(squeeze=64, expand=256, a1=II.ActivationLinearizer(initial_eq='relu'), a2=II.ActivationLinearizer(initial_eq='relu'), a3=II.ActivationLinearizer(initial_eq='relu')),
        layers.MaxPooling2D(3, 2),
        FireModule(squeeze=64, expand=256, a1=II.ActivationLinearizer(initial_eq='relu'), a2=II.ActivationLinearizer(initial_eq='relu'), a3=II.ActivationLinearizer(initial_eq='relu')),
        layers.Conv2D(1000, 1, strides=1),
        layers.AveragePooling2D(12, 1),
        II.ActivationLinearizer(initial_eq='relu'),#layers.Activation('relu'),
        layers.Flatten(),
        layers.Dense(10, activation='softmax') #Because of differing class size
    ])

    metrics_to_use = [TopKCategoricalAccuracy(name='T5'), TopKCategoricalAccuracy(name='T3', k=3), TopKCategoricalAccuracy(name='T2', k=2)]

#squeeze_net.summary()

print('\nTraining SqueezeNet')
squeeze_net.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=metrics_to_use)
squeeze_net.fit(train_data, epochs=5, validation_data=val_data)

print("\n")
then = time.time()
squeeze_net.evaluate(val_data)
print("Took: {}".format(str(int((time.time() - then) * 1000))))

#NOTE: All accuracy here is wrong because TopK instead of SparseTopK
#Control* ->  5.100s for 100% FVT5A,   0% FVT32A
#       NOTE:w/o Data-Augmentation

#Control -->  4.996s for 100% FVT3A,   0% FVT2A                        (Training is T5 ~95-100%   , T3 ~73-87%   , T2 ~46-66-65%)
#       NOTE: w/ Data-Augmentation


#1 PLUs --->  5.003s for 100% FVT5A,   0% FVT32A but 100% T2 Epoch 1-2 (Training is T5 ~61-54-58% , T3 ~25-44-33%, T2 ~10-30-22%)
#. . .
#4 PLUs --->  5.195s for 100% FVT5A,   0% FVT32A but 100% T2 Epoch 1-3 (Training is T5 ~56-53-58% , T3 ~31-19-39%, T2 ~20-14-29%)
#. . .
#16 PLUs -->  7.620s for   0% FVT5A,             but 100% T2 Epoch 1-3 (Training is T5 ~54-59%    , T3 ~30-43-41%, T2 ~19-10-27%)
#. . .
#25 PLUs --> 14.450s for 100% FVT5A,   0% FVT32A but 100% T2 Epoch 2   (Training is T5 ~48-72%    , T3 ~35-31-50%, T2 ~30-21-38%)


#1 ALs ---->  5.014s for 100% FVT5A,   0% FVT32A                       (Training is T5 ~69-100%   , T3 ~31-79%   , T2 ~27-46%   )
#. . .
#4 ALs ---->  5.042s for 100% FVT5A,   0% FVT32A                       (Training is T5 ~81-100%   , T3 ~35-82%   , T2 ~28-59%   )
#. . .
#16 ALs --->  5.320s for 100% FVT5A,   0% FVT32A                       (Training is T5 ~74-100%   , T3 ~33-83%   , T2 ~29-55%   )
#. . .
#25 ALs --->  6.210s for 100% FVT5A,   0% FVT32A                       (Training is T5 ~81-100%   , T3 ~37-80%   , T2 ~30-47%   )