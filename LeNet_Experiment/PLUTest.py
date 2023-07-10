import logging
import InferredActivations.Inferrer as IL
import LeNet_Experiment.Global as G
from keras import models, layers

ITModel = models.Sequential()
ITModel.add(layers.Conv2D(6, 5, activation=None, input_shape=G.x_train.shape[1:]))
ITModel.add(IL.PiecewiseLinearUnitV1())
ITModel.add(layers.AveragePooling2D(2))
ITModel.add(layers.Activation('sigmoid'))

ITModel.add(layers.Conv2D(16, 5, activation=None))
ITModel.add(IL.PiecewiseLinearUnitV1())
ITModel.add(layers.AveragePooling2D(2))
ITModel.add(layers.Activation('sigmoid'))

ITModel.add(layers.Conv2D(120, 5, activation=None))
ITModel.add(IL.PiecewiseLinearUnitV1())
ITModel.add(layers.Flatten())

ITModel.add(layers.Dense(84, activation=None))
ITModel.add(IL.PiecewiseLinearUnitV1())
ITModel.add(layers.Dense(10, activation='softmax'))
#ITModel.summary(line_length=150)

#First, collect stats
"""ITModel.layers[1].StatisticalAnalysisToggle(forceTo=True)
ITModel.layers[5].StatisticalAnalysisToggle(forceTo=True)
ITModel.layers[9].StatisticalAnalysisToggle(forceTo=True)
ITModel.layers[12].StatisticalAnalysisToggle(forceTo=True)

G.trainAndTime(ITModel, ee=2, eager=True)

ITModel.layers[1].StatisticalAnalysisToggle()
ITModel.layers[5].StatisticalAnalysisToggle()
ITModel.layers[9].StatisticalAnalysisToggle()
ITModel.layers[12].StatisticalAnalysisToggle()"""

IT_time, IT_history = G.trainAndTime(ITModel, ee=5)
logging.basicConfig(filename='infertanh.log', encoding='utf-8',level=logging.DEBUG)
logging.info('Time: ' + str(IT_time))
logging.info('Final Epoch Accuracy: ' + str(IT_history.history['val_accuracy'][4]))

#"""
ITModel.layers[1].Extract()
print("\n")
ITModel.layers[5].Extract()
print("\n")
ITModel.layers[9].Extract()
print("\n")
ITModel.layers[12].Extract()
#"""

#NOTE: gets around 95-98% accuracy after 5 epochs, really good. Very promising. Why is it better than simply linear equations?