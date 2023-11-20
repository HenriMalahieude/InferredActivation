#Taken from: Andrej Karpathy's GPT Video
#https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing#scrollTo=0e-Rbyr8sfM8

import math
import tensorflow as tf
import numpy as np

from Transformer import MultiHeadAttention
from keras import layers

# Hyperparameters
EPOCHS = 15 # How many times we cycle through the training data
BATCH_SIZE = 64 # How many independent sequences will we process in parallel?
ATTENT_SPAN = 32 # What is the maximum context length for predictions? (Used to build dataset too)
#VAL_SPLIT=0.1 # How much of the dataset to put to the side for validation
#EPOCH_STEPS = 100
N_EMBED = 64 
N_HEAD = 4 # Amount of Attention Heads in Multi-Headed Attention
N_LAYER = 4 # Amount of Multi-Headed Attention Layers (Transformers)
DROPOUT = 0.1
# ------------
DS_USAGE = 6 #To Be the Reduction (Denominator) against the dataset size
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
"""
print("Prepping Dataset")
dataset_full = None
DATASET_NAME = "shakespear_processed"

try:
    print("\tAttempting to Open Dataset")
    dataset_full = tf.data.Dataset.load(("./" + DATASET_NAME))

    print("\tChecking if Loaded Dataset can be used")
    can_use = True

    for x, y in dataset_full:
        if x.shape[-1] != ATTENT_SPAN: #Expecting a shape of B, T where T == ATTENT_SPAN
            print("Dataset is in incorrect format")
            raise ValueError("This dataset needs to be reformatted")
        break

except:
    print("\t(Re)building Dataset (nonexistent or incorrect format catch)")
    data_raw = tf.constant(encode(text))
    data_len = len(data_raw) // DS_USAGE
    data_input = []
    data_labels = []

    for i in range(data_len - ATTENT_SPAN): #Honestly, I should just load this into a generator function to be real
        print("\t% ", int((i / data_len) * 10000) / 100, "    ", end="\r")
        index_at = i + ATTENT_SPAN
        data_input.append(data_raw[i:index_at])
        data_labels.append(data_raw[index_at])

    print("\t% 100      ")

    print("\tInstancing Dataset")
    dataset_full = tf.data.Dataset.from_tensor_slices((data_input, data_labels))
    print("\tSaving Dataset")
    tf.data.Dataset.save(dataset_full, "./" + DATASET_NAME)

print("Batching and Prefetching Dataset")
#dataset_full.unbatch()
dataset_full = dataset_full.batch(BATCH_SIZE, drop_remainder=True)
dataset_full = dataset_full.prefetch(buffer_size=tf.data.AUTOTUNE)
text = None
"""
data_raw = tf.constant(encode(text))
class RGen(tf.keras.utils.Sequence):
    def __init__(self):
        super(RGen, self).__init__()

    def __len__(self):
        return math.ceil(len(data_raw) / ATTENT_SPAN) // BATCH_SIZE

    def __getitem__(self, idx):
        ix = tf.raw_ops.RandomUniformInt(shape=(BATCH_SIZE,), minval=0, maxval=len(data_raw) - BATCH_SIZE)
        x = tf.stack([data_raw[i:i+ATTENT_SPAN] for i in ix])
        y = tf.stack([data_raw[i+1:i+ATTENT_SPAN+1] for i in ix])
        return (x, y)

base_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
def custom_loss(y_true, y_pred): #Takes in logits of (B, T, vocab_size)
    shp = tf.shape(y_pred)

    y_pred = tf.reshape(y_pred, (shp[0] * shp[1], shp[2]))
    y_true = tf.reshape(y_true, (shp[0] * shp[1],))

    return base_loss(y_true, y_pred)

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
        #x = layers.Flatten()(x) # (B, T*C), which reduces the complexity of our model
        logits = self.lm_head(x) #(B, T, vocab_size) on normal; (B, vocab_size) with flatten
        #logits = tf.nn.softmax(logits)

        return logits

    def generate(self, idx, max_new_tokens): #Babble for me please
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -1 * ATTENT_SPAN:]
            # get the predictions
            logits = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] #(B, C)
            # apply softmax to get probabilities
            probs = tf.nn.softmax(logits, axis=-1) # (B, C)
            # sample from the distribution
            idx_next = tf.raw_ops.Multinomial(logits=probs, num_samples=1) # (B, 1)

            # append sampled index to the running sequence
            idx = tf.raw_ops.Concat(concat_dim=1, values=(idx, tf.cast(idx_next, dtype=tf.float32))) # (B, T+1)
        return idx
    
print("\nTraining")
#strat = tf.distribute.MirroredStrategy()
#print("\tAvailable Devices: {}".format(strat.num_replicas_in_sync))
#with strat.scope():
model = BigramLanguageModel()

model.compile(optimizer='adam', loss=custom_loss)
#print(model.call(tf.ones((BATCH_SIZE,ATTENT_SPAN), dtype=tf.float32)))
model.fit(RGen(), epochs=EPOCHS)

"""
print("\nShowing Predictor")
context = np.arange(ATTENT_SPAN)
context.shape = (1, ATTENT_SPAN)
result = model.call(tf.constant(context))
print("\t{}".format(result))
"""

print("\nGenerating")
context = tf.ones((1,ATTENT_SPAN), dtype=tf.float32)
print(decode(model.generate(context, max_new_tokens=500)[0].numpy()))
