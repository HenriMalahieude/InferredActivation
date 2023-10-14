#Taken from: Andrej Karpathy's GPT Video
#https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing#scrollTo=0e-Rbyr8sfM8

import tensorflow as tf
import numpy as np

from Transformer import MultiHeadAttention
from keras import layers

# Hyperparameters
EPOCHS = 1 # How many times we cycle through the training data
BATCH_SIZE = 64 # How many independent sequences will we process in parallel?
ATTENT_SPAN = 32 # What is the maximum context length for predictions? (Used to build dataset too)
VAL_SPLIT=0.1 # How much of the dataset to put to the side for validation
N_EMBED = 64 
N_HEAD = 4 # Amount of Attention Heads in Multi-Headed Attention
N_LAYER = 4 # Amount of Multi-Headed Attention Layers (Transformers)
DROPOUT = 0.1
# ------------
print("Starting NanoGPT Sandbox, credit to Andrej Karpathy's video")

print("\nOpening Shakespeare File")
with open('shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
print('\tCharacters: ' + ''.join(chars))
print('\tVocab Size: ', vocab_size)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

print("\nPrepping Dataset")
data_raw = tf.constant(encode(text))
label_raw = data_raw[ATTENT_SPAN::ATTENT_SPAN]
data_len = len(data_raw)

print("\tInstancing Dataset")
dataset_x = tf.data.Dataset.from_tensor_slices(data_raw)
dataset_y = tf.data.Dataset.from_tensor_slices(label_raw)

print("Batching and Prefetching Dataset")
#dataset_full.unbatch()
dataset_x = dataset_x.batch(ATTENT_SPAN, drop_remainder=True).batch(BATCH_SIZE, drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE) #Transform from () -> (T) -> (B, T)
dataset_y = dataset_y.batch(BATCH_SIZE, drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE)
dataset_full = tf.data.Dataset.zip(dataset_x, dataset_y)
text = None

print("Defining Transformer Stuff")
class TransformerBlock(layers.Layer):
    def __init__(self):
        super(TransformerBlock, self).__init__()
        head_size = N_EMBED // N_HEAD
        self.sa = MultiHeadAttention(head_count=N_HEAD, head_size=head_size)

        self.ffwd = tf.keras.models.Sequential([
            layers.Dense(4 * N_EMBED),
            layers.Activation("relu"), #<-------------------------- Place for approximations
            layers.Dense(N_EMBED),
            layers.Dropout(DROPOUT),
        ])
        
        self.ln1 = layers.LayerNormalization()
        self.ln2 = layers.LayerNormalization()

    def call(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(tf.keras.models.Model):
    def __init__(self):
        super(BigramLanguageModel, self).__init__()
        self.token_embedding_table = layers.Embedding(vocab_size, N_EMBED)
        self.position_embedding_table = layers.Embedding(ATTENT_SPAN, N_EMBED)
        self.blocks = tf.keras.models.Sequential([*[TransformerBlock() for _ in range(N_LAYER)]])
        self.ln_f = layers.LayerNormalization() # final layer norm
        self.lm_head = layers.Dense(vocab_size)

    def call(self, input): #Predict something
        T = tf.shape(input)[1]

        tok_emb = self.token_embedding_table(input) #(B, T, C)
        pos_emb = self.position_embedding_table(tf.range(T)) #(T, C)
        x = tok_emb + pos_emb #(B, T, C)
        x = self.blocks(x) #(B, T, C)
        x = self.ln_f(x) #(B, T, C)
        x = layers.Flatten()(x) # (B, T*C)
        x = self.lm_head(x) #(B, vocab_size)
        x = tf.nn.softmax(x)

        return x

    def generate(self, idx, max_new_tokens): #Babble for me please
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -1 * ATTENT_SPAN:]
            # get the predictions
            logits = self(idx_cond) # (B, vocab_size)
            # focus only on the last time step

            # apply softmax to get probabilities
            probs = tf.nn.softmax(logits, axis=-1) # (B, vocab_size)
            # sample from the distribution
            idx_next = tf.raw_ops.Multinomial(probs, 1) # (B, 1)

            # append sampled index to the running sequence
            idx = tf.raw_ops.Concat(1, (idx, idx_next)) # (B, T+1)
        return idx
    
print("\nTraining")
model = BigramLanguageModel()
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy())
model.fit(dataset_full, epochs=EPOCHS)

print("\nShowing Predictor")
context = np.arange(ATTENT_SPAN)
context.shape = (1, ATTENT_SPAN)
result = model.call(tf.constant(context))
print("\t{}".format(result))

print("\nGenerating")
context = tf.ones((1,1), dtype=tf.float32)
print(decode(model.generate(context, max_new_tokens=2000)[0].to_list()))
