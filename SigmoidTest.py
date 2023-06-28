import logging
import InferredActivations.Inferrer as IL
import InferredActivations.Inferrer.ActivationFunctions as IA
import LeNet_Experiment.Global as G
from keras import models, layers

logging.basicConfig(filename='infersigmoid.log', format='%(message)s', encoding='utf-8',level=logging.DEBUG)

ISModel = models.Sequential()
ISModel.add(layers.Conv2D(6, 5, input_shape=G.x_train.shape[1:]))
ISModel.add(layers.Activation('tanh'))
ISModel.add(layers.AveragePooling2D(2))
ISModel.add(IL.InferredActivation(eq_funcs=IA.NewNewSigApproximator)) 

ISModel.add(layers.Conv2D(16, 5))
ISModel.add(layers.Activation('tanh'))
ISModel.add(layers.AveragePooling2D(2))
ISModel.add(layers.Activation('sigmoid'))#IL.InferredActivation(eq_funcs=IA.SigApproximator))

ISModel.add(layers.Conv2D(120, 5))
ISModel.add(layers.Activation('tanh'))
ISModel.add(layers.Flatten())

ISModel.add(layers.Dense(84))
ISModel.add(layers.Activation('tanh'))
ISModel.add(layers.Dense(10, activation='softmax'))

#ISQModel = G.quantizeBeforeTraining(ISModel)

ISModel.summary(print_fn = logging.info, line_length = 150)

#NOTE: At some point I need to figure out how to get the gradients to not be infinitesimally small when printing, what a pain
#ISModel.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])
#with tf.GradientTape() as tape:
#    y = ISModel(G.x_train[:0])
#    loss = losses.sparse_categorical_crossentropy(y_true = G.y_train[:0], y_pred = y)
#    gradients = tape.gradient(loss, ISModel.trainable_weights)
#    print(gradients[2] * 10000, gradients[5])
#print("\n")

IS_time, IS_history = G.trainAndTime(ISModel)
logging.info('Normal Time: ' + str(IS_time) + "; Normal Final Accuracy: " + str(IS_history.history['val_accuracy'][4]))

#ISQ_time, ISQ_history = G.trainAndTime(ISQModel)
#logging.info('Quantized Time: ' + str(ISQ_time) + "; Quantized Final Accuracy: " + str(ISQ_history.history['val_accuracy'][4]))

#"""
ISModel.layers[3].Extract()
print("\n")
#ISModel.layers[7].Extract()
#"""