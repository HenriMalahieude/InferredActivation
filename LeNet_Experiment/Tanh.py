import logging
import InferredActivations.Inferrer as IL
import InferredActivations.Inferrer.ActivationFunctions as IA
import Global as G
from keras import models, layers

ITModel = models.Sequential()
ITModel.add(layers.Conv2D(6, 5, activation=None, input_shape=G.x_train.shape[1:]))
ITModel.add(IL.InferredActivation(eq_funcs=IA.TanhApproximator))
ITModel.add(layers.AveragePooling2D(2))
ITModel.add(layers.Activation('sigmoid'))

ITModel.add(layers.Conv2D(16, 5, activation=None))
ITModel.add(IL.InferredActivation(eq_funcs=IA.TanhApproximator))
ITModel.add(layers.AveragePooling2D(2))
ITModel.add(layers.Activation('sigmoid'))

ITModel.add(layers.Conv2D(120, 5, activation=None))
ITModel.add(IL.InferredActivation(eq_funcs=IA.TanhApproximator))
ITModel.add(layers.Flatten())

ITModel.add(layers.Dense(84, activation=None))
ITModel.add(IL.InferredActivation(eq_funcs=IA.TanhApproximator))
ITModel.add(layers.Dense(10, activation='softmax'))
#ITModel.summary(line_length=150)

IT_time, IT_history = G.trainAndTime(ITModel)
logging.basicConfig(filename='infertanh.log', encoding='utf-8',level=logging.DEBUG)
logging.info('Time: ' + str(IT_time))
logging.info('Final Epoch Accuracy: ' + str(IT_history.history['val_accuracy'][4]))

"""
ITModel.layers[1].Extract()
print("\n")
ITModel.layers[5].Extract()
print("\n")
ITModel.layers[9].Extract()
print("\n")
ITModel.layers[12].Extract()
#"""