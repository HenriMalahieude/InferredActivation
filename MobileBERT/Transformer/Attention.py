import tensorflow as tf
import numpy as np

from keras import layers

TEMP_INF = 1000000000
#A lot of help from Andrej Karpathy's GPT video: https://www.youtube.com/watch?v=kCc8FmEb1nY

class SingleHeadAttention(tf.keras.layers.Layer):
    def __init__(self, 
                 encoder: bool = False, #either encode (make text) or decode (understand text)
                 head_size: int = 16,
                 softmax_replacement = tf.nn.softmax 
                ):
        super(SingleHeadAttention, self).__init__()
        self.encoder = encoder
        self.head_size = head_size
        self.softmax_func = softmax_replacement
    
    #Expects something like (Batch, Time, Channel)
    def build(self, input_shape):
        B, T, C = input_shape
        tril = np.tril(tf.ones((T, T)))
        inv_tril = tf.cast(tf.equal(tril, 0), tf.float32)
        self.inv_tril_inf = inv_tril * (TEMP_INF * -1)

        #Key: I am this token
        self.key = layers.Dense(self.head_size, use_bias=False) 
        self.key.build(input_shape)

        #Query: I am looking for this type of token
        self.query = layers.Dense(self.head_size, use_bias=False) 
        self.query.build(input_shape)

        #Value: What token this head will mask/focus on
        self.value = layers.Dense(self.head_size, use_bias=False)
        self.value.build(input_shape)
        #super(SingleHeadAttention, self).build(input_shape)

    def call(self, inputs):
        #B, T, C = inputs.shape
        k = self.key(inputs) # [B, T, head]
        q = self.query(inputs) # [B, T, head]
        wei = tf.linalg.matmul(q, tf.transpose(k, perm=[0, 2, 1])) / (self.head_size ** 0.5) # [B, T, head] x [B, head, T] = [B, T, T] / sqrt(d_k)

        #wei = tf.constant(0, shape=(T, T))
        if not self.encoder:
            wei += self.inv_tril_inf # [B, T, T] + [T, T] -> tf assumes: [B, T, T] + [B, T, T]

        # V--------------------------------------------------------------------This is where we could potentially do "approximations"
        wei = self.softmax_func(wei) 
        
        v = self.value(inputs)
        y = tf.linalg.matmul(wei, v)#inputs)

        return y
    
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self,
                 encoder = False,
                 head_count: int = 3,
                 head_size: int = 16,
                 softmax_replacement = tf.nn.softmax
                 ):
        super(MultiHeadAttention, self).__init__()

        self.heads = []
        for i in range(head_count):
            self.heads.append(SingleHeadAttention(encoder, head_size, softmax_replacement))
    
    def build(self, input_shape):
        for i in range(len(self.heads)):
            self.heads[i].build(input_shape)

    def call(self, inputs):
        vals = []
        for i in range(len(self.heads)):
            val = self.heads[i].call(inputs)
            vals.append(val)

        final = tf.concat(vals, axis=-1) #So for example 4x (16, 32, 16) turns into (16, 32, 64)

        return final