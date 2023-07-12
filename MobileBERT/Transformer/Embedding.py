import tensorflow as tf

#Shamelessly Stolen from:
#https://sungwookyoo.github.io/tips/CompareTensorflowAndPytorch/#big-difference-tensorflow-vs-pytorch
class Embedding(tf.keras.layers.Layer):
  
    def __init__(self, input_dim, output_dim, padding_idx=0, **kwargs):
        """ default padding_idx=0.
        
        Call Args:
            inputs: [B, T]
        
        description:
            input_dim: V (vocabulary size)
            output_dim: D 
        """
        super(Embedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.padding_idx = padding_idx

    def build(self, input_shape):
        self.embeddings = self.add_weight(
          shape=(self.input_dim, self.output_dim),
          initializer='random_normal',
          dtype='float32')

    def call(self, inputs): 
        def compute_mask():
            return tf.not_equal(inputs, self.padding_idx)
        
        out = tf.nn.embedding_lookup(self.embeddings, inputs)
        masking = compute_mask() # [B, T], bool
        masking = tf.cast(tf.tile(masking[:,:, tf.newaxis], [1,1,self.output_dim]), 
                          dtype=tf.float32) # [B, T, D]
        return tf.multiply(out, masking)
