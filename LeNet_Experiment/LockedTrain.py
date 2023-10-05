import logging
import InferredActivations.Inferrer as IL
import InferredActivations.Inferrer.ProtoActivations as IA
import LeNet_Experiment.Global as G
from keras import models, layers, losses

logging.basicConfig(filename='infer_quantizedall.log', encoding='utf-8',level=logging.DEBUG)

CModel = models.load_model("LeNetControl")

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

layerMapping = [[0, 0], [2, 2], [4, 4], [6, 6], [8, 8], [10, 10], [11, 11], [13, 13]]
for i in range(len(layerMapping)):
        c_lay = layerMapping[i][0]
        b_lay = layerMapping[i][1]
        IAModel.layers[c_lay].set_weights(CModel.layers[b_lay].get_weights())
        IAModel.layers[c_lay].trainable = False

IAModel.summary(print_fn=logging.info)

IAModel.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])
IA_History = IAModel.fit(G.x_train, G.y_train, batch_size=64, epochs=15, validation_data=(G.x_val, G.y_val))

#logging.info('Time: ' + str(IA_Time))
logging.info('Final Epoch Accuracy: ' + str(IA_History.history['val_accuracy'][4]))