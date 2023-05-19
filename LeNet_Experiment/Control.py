import logging
import Global as G
from keras import models, layers

les_control = models.Sequential()
les_control.add(layers.Conv2D(6, 5, input_shape=G.x_train.shape[1:]))
les_control.add(layers.Activation('tanh'))
les_control.add(layers.AveragePooling2D(2))
les_control.add(layers.Activation('sigmoid'))

les_control.add(layers.Conv2D(16, 5))
les_control.add(layers.Activation('tanh'))
les_control.add(layers.AveragePooling2D(2))
les_control.add(layers.Activation('sigmoid'))

les_control.add(layers.Conv2D(120, 5))
les_control.add(layers.Activation('tanh'))
les_control.add(layers.Flatten())

les_control.add(layers.Dense(84))
les_control.add(layers.Activation('tanh'))
les_control.add(layers.Dense(10, activation='softmax'))

control_time, control_history = G.trainAndTime(les_control)

#print(control_time, control_history.history['val_accuracy'][4])
logging.basicConfig(filename='control.log', encoding='utf-8',level=logging.DEBUG)
logging.info('Time: ' + str(control_time))
logging.info('Final Epoch Accuracy: ' + str(control_history.history['val_accuracy'][4]))

#So external programs can use it
les_control.save("./LeNetControl")