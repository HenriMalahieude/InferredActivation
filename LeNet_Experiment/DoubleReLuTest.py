import logging
import InferredActivations.Inferrer as IL
import InferredActivations.Inferrer.ProtoActivations as IA
import LeNet_Experiment.Global as G
from keras import models, layers

logging.basicConfig(filename='infer_doublereluall.log', encoding='utf-8',level=logging.DEBUG)

Model = models.Sequential()
Model.add(layers.Conv2D(6, 5, activation=None, input_shape=G.x_train.shape[1:]))
Model.add(IL.InferredActivation(eq_funcs=IA.DoubleReLUApproximator))
Model.add(layers.AveragePooling2D(2))
Model.add(layers.Activation('sigmoid'))

Model.add(layers.Conv2D(16, 5, activation=None))
Model.add(layers.Activation('tanh'))
Model.add(layers.AveragePooling2D(2))
Model.add(layers.Activation('sigmoid'))

Model.add(layers.Conv2D(120, 5, activation=None))
Model.add(layers.Activation('tanh'))
Model.add(layers.Flatten())

Model.add(layers.Dense(84, activation=None))
Model.add(layers.Activation('tanh'))#IL.InferredActivation(eq_funcs=IA.DoubleReLUApproximator))
Model.add(layers.Dense(10, activation='softmax'))

Model.summary()

Time, History = G.trainAndTime(Model)

logging.info('Time: ' + str(Time))
logging.info('Final Epoch Accuracy: ' + str(History.history['val_accuracy'][4]))

Model.layers[1].Extract()

#NOTE: Final average accuracy is around 0.1070 no matter what happens. So.... it's pretty bad lol