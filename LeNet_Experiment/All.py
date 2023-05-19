import logging
import InferredActivations.Inferrer as IL
import InferredActivations.Inferrer.ActivationFunctions as IA
import Global as G
from keras import models, layers

IAModel = models.Sequential()
IAModel.add(layers.Conv2D(6, 5, activation=None, input_shape=G.x_train.shape[1:]))
IAModel.add(IL.InferredActivation(eq_funcs=IA.TanhApproximator))
IAModel.add(layers.AveragePooling2D(2))
IAModel.add(IL.InferredActivation()) #Defaults to Sigmoid

IAModel.add(layers.Conv2D(16, 5, activation=None))
IAModel.add(IL.InferredActivation(eq_funcs=IA.TanhApproximator))
IAModel.add(layers.AveragePooling2D(2))
IAModel.add(IL.InferredActivation())

IAModel.add(layers.Conv2D(120, 5, activation=None))
IAModel.add(IL.InferredActivation(eq_funcs=IA.TanhApproximator))
IAModel.add(layers.Flatten())

IAModel.add(layers.Dense(84, activation=None))
IAModel.add(IL.InferredActivation(eq_funcs=IA.TanhApproximator))
IAModel.add(layers.Dense(10, activation='softmax'))

IATime, IAHistory = G.trainAndTime(IAModel)
logging.basicConfig(filename='inferall.log', encoding='utf-8',level=logging.DEBUG)
logging.info('Time: ' + str(IATime))
logging.info('Final Epoch Accuracy: ' + str(IAHistory.history['val_accuracy'][4]))