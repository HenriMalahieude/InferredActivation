#Taken from: Andrej Karpathy's GPT Video
#https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing#scrollTo=0e-Rbyr8sfM8

import tensorflow as tf
import numpy as np

from Transformer import MultiHeadAttention
from keras import layers

# hyperparameters
BATCH_SIZE = 16 # how many independent sequences will we process in parallel?
ATTENT_SPAN = 32 # what is the maximum context length for predictions?
VAL_SPLIT=0.1
N_EMBED = 64
N_HEAD = 4 #Attention Heads
N_LAYER = 4
DROPOUT = 0.0

DEFAULT_DATASET_FILENAME = "nano_shakespeare_ds"
# ------------

#Get Input Data
with open('shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
print('Characters: ' + ''.join(chars))
print('Vocab Size: ', vocab_size)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

print("Prepping Dataset (%)")
data_raw = tf.constant(encode(text))
data_len = len(data_raw) // 4
data_input = []
data_labels = []

text = None
dataset_full = None

try:
    dataset_full = tf.data.Dataset.load(("./" + DEFAULT_DATASET_FILENAME))
except:
    print("\tFormatting Dataset")

    for i in range(data_len - ATTENT_SPAN):
        print("\t% ", int((i / data_len) * 10000) / 100, end="\r")
        index_at = i + ATTENT_SPAN
        data_input.append(data_raw[i:index_at])
        data_labels.append(data_raw[index_at])

    print("\t% 100    ")

    print("Creating Dataset")
    dataset_full = tf.data.Dataset.from_tensor_slices((data_input, data_labels))
    tf.data.Dataset.save(dataset_full, "./" + DEFAULT_DATASET_FILENAME)

print("Dataset Element Size: {}".format(dataset_full.cardinality()))

print("Batching and Prefetching Dataset")
#dataset_full.unbatch()
dataset_full = dataset_full.batch(BATCH_SIZE, drop_remainder=True)
dataset_full = dataset_full.prefetch(buffer_size=tf.data.AUTOTUNE)

print("Checking Dataset")
for x, y in dataset_full:
    print("\tInput:", x.shape)
    print("\tOutput:", y.shape)
    break

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
        self.ln_f = layers.LayerNormalization(N_EMBED) # final layer norm
        self.lm_head = layers.Dense(vocab_size)

    def build(self, input_shape):
        self.lm_head.build(N_EMBED)

    def call(self, input):
        assert len(input.shape) <= 2
        assert len(input.shape) > 0

        if len(input.shape) == 2:
            B, T = input.shape
        elif len(input.shape) == 1:
            T = input.shape[0]

        tok_emb = self.token_embedding_table(input)
        pos_emb = self.position_embedding_table(np.arange(T))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -1 * ATTENT_SPAN:]
            # get the predictions
            logits = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = tf.nn.softmax(logits, axis=-1) # (B, C)
            # sample from the distribution
            idx_next = tf.raw_ops.Multinomial(probs, 1) # (B, 1)

            # append sampled index to the running sequence
            idx = tf.raw_ops.Concat(1, (idx, idx_next)) # (B, T+1)
        return idx
    
print("Training")
model = BigramLanguageModel()
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
model.fit(dataset_full, epochs=10)

print("Generation")
context = tf.ones((1,1), dtype=tf.float32)
print(decode(model.generate(context, max_new_tokens=2000)[0].to_list()))
