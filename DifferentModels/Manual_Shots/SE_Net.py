import tensorflow as tf
import tensorflow_datasets as tfds
import time
import InferredActivations.Inferrer as II
from keras import models, layers
from keras.metrics import TopKCategoricalAccuracy

BATCH_SIZE = 32
IMAGE_SIZE = (224, 224, 3)
REDUCTION_RATIO = 4
AUGMENT_DATA = True

print("Starting SE_Net Sandbox")
print("\tStats:\n\t\tImage Size of {}\n\t\tReduction Ratio of {}".format(IMAGE_SIZE, REDUCTION_RATIO))

print("\nLoading in imagenette/320px-v2")
train_ds, val_ds = tfds.load("imagenette/320px-v2", split=['train', 'validation'], as_supervised=True, batch_size=BATCH_SIZE)

resize_rescale = models.Sequential([
    layers.Resizing(IMAGE_SIZE[0], IMAGE_SIZE[1]),
    layers.Rescaling(1./255)
])

def preprocess1(x, y):
        return resize_rescale(x, training=True), y

train_data = train_ds.map(preprocess1)
val_data = val_ds.map(preprocess1)

if AUGMENT_DATA:
    print("\nAugmenting Data")
    augmenter = models.Sequential([
        layers.RandomRotation(factor=(-0.05, 0.05), fill_mode='constant'),
        layers.Resizing(IMAGE_SIZE[0]*3, IMAGE_SIZE[1]*3),
        layers.RandomFlip(),
        layers.RandomZoom((0, -0.66), fill_mode='constant'),
        resize_rescale
    ])

    def preprocess2(x, y):
        return augmenter(x, training=True), y
    
    train_data = train_data.concatenate(train_ds.map(preprocess2))
    print("\tDoubled the training size")

print("\nDefining Squeeze_Excitation_Block")
#Taken from https://github.com/taki0112/SENet-Tensorflow/tree/master as source
class Squeeze_Excitation_Block(layers.Layer):
    def __init__(self, ratio,
                 a1 = layers.Activation('relu'),
                 a2 = layers.Activation('sigmoid')):
        super(Squeeze_Excitation_Block, self).__init__()
        self.ratio = ratio
        self.activation1 = a1
        self.activation2 = a2
    
    def build(self, input_shape):
        print(input_shape)
        self.excitation = models.Sequential([
            layers.GlobalAveragePooling2D(input_shape=input_shape), #This entire thing doesn't work btw, something to do WITH INPUT SHAPES BEING WRONG ALL THE TIME GODDAMN IT

            layers.Dense(units=(input_shape[-1]/self.ratio), use_bias=False), #Rest of this is the "excitation layer"
            self.activation1,
            layers.Dense(units=input_shape[-1], use_bias=False),
            self.activation2,
            layers.Reshape([1, 1, input_shape[-1]]) #So we can multiply to the input
        ])
    
    def call(self, input):
        return input * self.excitation(input) #Otherwise known as "The Scale"

print("\nDefining SE_ResidualBlock")
class SE_ResidualBlock(layers.Layer):
    def __init__(self, 
                 filters=(64,64,256),
                 stride=1,

                 #Residual Activations
                 a1 = layers.Activation("relu"),
                 a2 = layers.Activation("relu"),
                 a3 = layers.Activation("relu"),
                 
                 #Squeeze-Excitation Activations
                 ase1 = layers.Activation('relu'),
                 ase2 = layers.Activation('sigmoid')):
        super(SE_ResidualBlock, self).__init__()
        self.filters = filters
        self.stride = stride
        self.activation1 = a1
        self.activation2 = a2
        self.activation3 = a3
        self.activation_se1 = ase1
        self.activation_se2 = ase2
    
    def build(self, input_shape):
        f1, f2, f3 = self.filters

        self.normal_pass = models.Sequential([
            layers.Conv2D(f1, 1, strides=self.stride),
            layers.BatchNormalization(),
            self.activation1,

            layers.Conv2D(f2, 3, padding='same'),
            layers.BatchNormalization(),
            self.activation2,

            layers.Conv2D(f3, 1),
            layers.BatchNormalization(),
        ])

        self.excitation = models.Sequential([
            layers.GlobalAveragePooling2D(),

            layers.Dense(units=(f3/REDUCTION_RATIO), use_bias=False), #Rest of this is the "excitation layer"
            self.activation_se1,
            layers.Dense(units=f3, use_bias=False),
            self.activation_se2,
            layers.Reshape([1, 1, f3]) #So we can multiply to the input
        ])

        channel_increase = not (input_shape[-1] == f3)
        if channel_increase:
            self.id_pass = tf.keras.models.Sequential([
                layers.Conv2D(f3, 1, strides=self.stride),
                layers.BatchNormalization(),
            ])
        else:
            self.id_pass = layers.BatchNormalization()

        self.final_pass = self.activation3

    def call(self, input):
        y = self.normal_pass(input)
        #print(y.shape)
        yy = self.excitation(y)#bc this works better sighhhhh
        #print(yy.shape)

        return self.final_pass((y * yy) + self.id_pass(input))
    
print('\nStarting up Multi-GPU training sequence')
strat = tf.distribute.MirroredStrategy()
print('\tDevices available: {}'.format(strat.num_replicas_in_sync))

with strat.scope():
    se_resnet_50 = models.Sequential([
        layers.Conv2D(64, 7, strides=2, input_shape=IMAGE_SIZE),
        layers.BatchNormalization(),
        II.ActivationLinearizer(initial_eq='relu'),#layers.Activation('relu'),
        layers.MaxPooling2D(3, 2),

        SE_ResidualBlock((64, 64, 256), a1=II.ActivationLinearizer(initial_eq='relu'), a2=II.ActivationLinearizer(initial_eq='relu'), a3=II.ActivationLinearizer(initial_eq='relu'), ase1=II.ActivationLinearizer(initial_eq='relu'), ase2=II.ActivationLinearizer(initial_eq='sigmoid')),
        SE_ResidualBlock((64, 64, 256), a1=II.ActivationLinearizer(initial_eq='relu'), a2=II.ActivationLinearizer(initial_eq='relu'), a3=II.ActivationLinearizer(initial_eq='relu'), ase1=II.ActivationLinearizer(initial_eq='relu'), ase2=II.ActivationLinearizer(initial_eq='sigmoid')),
        SE_ResidualBlock((64, 64, 256), a1=II.ActivationLinearizer(initial_eq='relu'), a2=II.ActivationLinearizer(initial_eq='relu'), a3=II.ActivationLinearizer(initial_eq='relu'), ase1=II.ActivationLinearizer(initial_eq='relu'), ase2=II.ActivationLinearizer(initial_eq='sigmoid')),

        SE_ResidualBlock((128, 128, 512), stride=2, a1=II.ActivationLinearizer(initial_eq='relu'), a2=II.ActivationLinearizer(initial_eq='relu'), a3=II.ActivationLinearizer(initial_eq='relu'), ase1=II.ActivationLinearizer(initial_eq='relu'), ase2=II.ActivationLinearizer(initial_eq='sigmoid')),
        SE_ResidualBlock((128, 128, 512), a1=II.ActivationLinearizer(initial_eq='relu'), a2=II.ActivationLinearizer(initial_eq='relu'), a3=II.ActivationLinearizer(initial_eq='relu'), ase1=II.ActivationLinearizer(initial_eq='relu'), ase2=II.ActivationLinearizer(initial_eq='sigmoid')),
        SE_ResidualBlock((128, 128, 512), a1=II.ActivationLinearizer(initial_eq='relu'), a2=II.ActivationLinearizer(initial_eq='relu'), a3=II.ActivationLinearizer(initial_eq='relu'), ase1=II.ActivationLinearizer(initial_eq='relu'), ase2=II.ActivationLinearizer(initial_eq='sigmoid')),

        SE_ResidualBlock((256, 256, 1024), stride=2, a1=II.ActivationLinearizer(initial_eq='relu'), a2=II.ActivationLinearizer(initial_eq='relu'), a3=II.ActivationLinearizer(initial_eq='relu'), ase1=II.ActivationLinearizer(initial_eq='relu'), ase2=II.ActivationLinearizer(initial_eq='sigmoid')),
        SE_ResidualBlock((256, 256, 1024), a1=II.ActivationLinearizer(initial_eq='relu'), a2=II.ActivationLinearizer(initial_eq='relu'), a3=II.ActivationLinearizer(initial_eq='relu'), ase1=II.ActivationLinearizer(initial_eq='relu'), ase2=II.ActivationLinearizer(initial_eq='sigmoid')),
        SE_ResidualBlock((256, 256, 1024), a1=II.ActivationLinearizer(initial_eq='relu'), a2=II.ActivationLinearizer(initial_eq='relu'), a3=II.ActivationLinearizer(initial_eq='relu'), ase1=II.ActivationLinearizer(initial_eq='relu'), ase2=II.ActivationLinearizer(initial_eq='sigmoid')),
        SE_ResidualBlock((256, 256, 1024), a1=II.ActivationLinearizer(initial_eq='relu'), a2=II.ActivationLinearizer(initial_eq='relu'), a3=II.ActivationLinearizer(initial_eq='relu'), ase1=II.ActivationLinearizer(initial_eq='relu'), ase2=II.ActivationLinearizer(initial_eq='sigmoid')),
        SE_ResidualBlock((256, 256, 1024), a1=II.ActivationLinearizer(initial_eq='relu'), a2=II.ActivationLinearizer(initial_eq='relu'), a3=II.ActivationLinearizer(initial_eq='relu'), ase1=II.ActivationLinearizer(initial_eq='relu'), ase2=II.ActivationLinearizer(initial_eq='sigmoid')),
        SE_ResidualBlock((256, 256, 1024), a1=II.ActivationLinearizer(initial_eq='relu'), a2=II.ActivationLinearizer(initial_eq='relu'), a3=II.ActivationLinearizer(initial_eq='relu'), ase1=II.ActivationLinearizer(initial_eq='relu'), ase2=II.ActivationLinearizer(initial_eq='sigmoid')),

        SE_ResidualBlock((512, 512, 2048), stride=2, a1=II.ActivationLinearizer(initial_eq='relu'), a2=II.ActivationLinearizer(initial_eq='relu'), a3=II.ActivationLinearizer(initial_eq='relu'), ase1=II.ActivationLinearizer(initial_eq='relu'), ase2=II.ActivationLinearizer(initial_eq='sigmoid')),
        SE_ResidualBlock((512, 512, 2048), a1=II.ActivationLinearizer(initial_eq='relu'), a2=II.ActivationLinearizer(initial_eq='relu'), a3=II.ActivationLinearizer(initial_eq='relu'), ase1=II.ActivationLinearizer(initial_eq='relu'), ase2=II.ActivationLinearizer(initial_eq='sigmoid')),
        SE_ResidualBlock((512, 512, 2048), a1=II.ActivationLinearizer(initial_eq='relu'), a2=II.ActivationLinearizer(initial_eq='relu'), a3=II.ActivationLinearizer(initial_eq='relu'), ase1=II.ActivationLinearizer(initial_eq='relu'), ase2=II.ActivationLinearizer(initial_eq='sigmoid')),

        layers.AveragePooling2D(2, padding='same'),
        layers.Flatten(),

        layers.Dense(256),
        II.ActivationLinearizer(initial_eq='relu'),#layers.Activation('relu'),

        layers.Dense(128),
        II.ActivationLinearizer(initial_eq='relu'),#layers.Activation('relu'),

        layers.Dense(10, activation='softmax')
    ])

    metrics_to_use = [TopKCategoricalAccuracy(name='T5'), TopKCategoricalAccuracy(name='T3', k=3), TopKCategoricalAccuracy(name='T2', k=2)]

#se_resnet_50.summary()

#"""
print("\nStarting Training:")
se_resnet_50.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=metrics_to_use)
se_resnet_50.fit(train_data, epochs=5, validation_data=val_data)

print("\n")
then = time.time()
se_resnet_50.evaluate(val_data)
print("Took {}ms".format(int((time.time() - then) * 1000)))
#"""

#Control ->  5.266s for 31% 21% 17%

#NOTE: sometimes getting nan val_loss
#17PLUs -->  6.487s for 31%  3%  0.1%
#35PLUs --> 15.369s for 19% 10%  6% (Best Epoch-2 92% 92% 92%)
#66PLUs --> 47.449s fpr 93% 92%  3.6% (Best Epoch-3 98% 98% 96%)

#17ALs --->  5.306s for 19% 12%  7% (Best Epoch-3 87% 56% 31%)
#35ALs --->  6.375s for 25% 14%  8% (Best Epoch-5)
#66ALs ---> 13.908s for 38% 23% 14% (Best Epoch-5)