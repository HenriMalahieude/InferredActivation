import numpy as np
import InferredActivations.Inferrer as IAI
import InferredActivations.Inferrer.ProtoActivations as AF
import InferredActivations.Inferrer.ActivationLinearizer as AL
import LeNet_Experiment.Global as G
import tensorflow as tf

#NOTE: Using the black box method, where only the output from teacher is used

teacher = tf.keras.models.Sequential()
teacher.add(tf.keras.layers.Conv2D(6, 5, input_shape=G.x_train.shape[1:]))
teacher.add(tf.keras.layers.Activation('tanh'))
teacher.add(tf.keras.layers.AveragePooling2D(2))
teacher.add(tf.keras.layers.Activation('sigmoid'))

teacher.add(tf.keras.layers.Conv2D(16, 5))
teacher.add(tf.keras.layers.Activation('tanh'))
teacher.add(tf.keras.layers.AveragePooling2D(2))
teacher.add(tf.keras.layers.Activation('sigmoid'))

teacher.add(tf.keras.layers.Conv2D(120, 5))
teacher.add(tf.keras.layers.Activation('tanh'))
teacher.add(tf.keras.layers.Flatten())

teacher.add(tf.keras.layers.Dense(84))
teacher.add(tf.keras.layers.Activation('tanh'))
teacher.add(tf.keras.layers.Dense(10, activation='softmax')) #We'll need to pop this for logits 

print("\n\tStarting teacher training")
teach_tm, teach_hs = G.trainAndTime(teacher, ee=3)

print("\n\tProducing teacher output")
teacher_predict = teacher.predict(G.x_train, batch_size=64)
print("\tGot:", teacher_predict.shape)
#print("\t\tExample 0:", teacher_predict[0])

#Formatting time, if we want to use SparseCategoricalCrossentropy in base func
"""new_training_y = []
for i in range(teacher_predict.shape[0]):
    big = -1
    c_ind = 0
    for j in range(teacher_predict.shape[1]):
        if teacher_predict[i][j] > big:
            big = teacher_predict[i][j]
            c_ind = j
    new_training_y.append(c_ind)

new_training_y = np.array(new_training_y)
print("\n\tReformatted to:", new_training_y.shape)"""

#From: https://arxiv.org/pdf/1503.02531.pdf
class DistillationLoss(tf.keras.losses.Loss):
    def __init__(self):
        super(DistillationLoss, self).__init__()
        self.base = tf.keras.losses.CategoricalCrossentropy()

    def call(self, y_true, y_pred):
        N = tf.cast(tf.size(y_pred), dtype=y_pred.dtype)
        TEMP = 5
        extra_loss_tensor = (1 / (N * (TEMP ** 2))) * (y_pred - tf.cast(y_true, y_pred.dtype))
        base_loss = self.base(y_true, y_pred)
        return base_loss + tf.reduce_sum(extra_loss_tensor)
    
student = tf.keras.models.Sequential()
student.add(tf.keras.layers.Conv2D(6, 5, input_shape=G.x_train.shape[1:]))
student.add(tf.keras.layers.Activation('tanh'))
student.add(tf.keras.layers.AveragePooling2D(2))
student.add(AL.ActivationLinearizer())

student.add(tf.keras.layers.Conv2D(16, 5))
student.add(tf.keras.layers.Activation('tanh'))
student.add(tf.keras.layers.AveragePooling2D(2))
student.add(AL.ActivationLinearizer())

student.add(tf.keras.layers.Conv2D(120, 5))
student.add(tf.keras.layers.Activation('tanh'))
student.add(tf.keras.layers.Flatten())

student.add(tf.keras.layers.Dense(84))
student.add(tf.keras.layers.Activation('tanh'))
student.add(tf.keras.layers.Dense(10, activation='softmax'))
#student.add(tf.keras.layers.Activation('softmax'))

print("\n\tStarting student training on teacher predictions: (Specialized Loss Func)") #Currently BLACK BOX like, based on output only
student.compile(optimizer='adam', loss=DistillationLoss(), metrics=["accuracy"])

student.fit(G.x_train, teacher_predict, batch_size=64, epochs=5)

print("\n\tEvaluating Success: (Base Loss Func)")
student.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
student.evaluate(x=G.x_train, y=G.y_train, batch_size=64)
student.evaluate(x=G.x_val, y=G.y_val, batch_size=64)
student.evaluate(x=G.x_test, y=G.y_test, batch_size=64)

"""
Replacing all Activations with PLU results in a learned 92% (sometimes beating the teacher) accuracy on training data, 
    and a 0-1% validation accuracy hit compared to teacher. Though, only reaches 97% teacher accuracy during training

Cannot replace more than the 2 tanh with their approximations before loss is over 80%

Attempting to use NewNewSigmoid is pretty unimpressive, 10% accuracy

ActivationLinearizer, the generic approach, gets results that beat the teacher sometimes
"""