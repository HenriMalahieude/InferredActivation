#Taken from: Andrej Karpathy's GPT Video
#https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing#scrollTo=0e-Rbyr8sfM8

import tensorflow as tf
import Transformer.Linear as L
import Transformer as T
import numpy as np

# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0
# ------------

#Get Input Data
with open('shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
print('Characters: '.join(chars))
print('Vocab Size: ', vocab_size)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string


data = tf.constant(encode(text), dtype=tf.int32)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

#TODO: Check this stuff
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = np.randint(high=len(data) - block_size, size=(batch_size,))
    x = tf.stack([data[i:i+block_size] for i in ix])
    y = tf.stack([data[i+1:i+block_size+1] for i in ix])
    #x, y = x.to(device), y.to(device)
    return x, y

class FeedForward(tf.keras.layers.Layer):
    def __init__(self):
        self.net = tf.keras.models.Sequential()
        self.net.add(L.Linear(n_embd, 4 * n_embd))
        self.net.add(tf.keras.layers.Activation("relu"))
        self.net.add(L.Linear(4 * n_embd, n_embd))
        self.net.add(tf.keras.layers.Dropout(dropout))
    
    def build(self, input_shape):
        self.net.layers[0].build([1, n_embd])
        self.net.layers[2].build([1, 4 * n_embd])

    def call(self, input):
        return self.net.call(input)
    
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self):
        super(TransformerBlock, self).__init__()
        head_size = n_embd // n_head
        self.sa = T.MultiHeadAttention(head_count=n_head, head_size=head_size)
        self.ffwd = FeedForward()
        self.ln1 = tf.keras.layers.LayerNormalization()
        self.ln2 = tf.keras.Layers.LayerNormalization()

    def call(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(tf.keras.models.Model):
    def __init__(self):
        super(BigramLanguageModel, self).__init__()
        self.token_embedding_table = tf.keras.layers.Embedding(vocab_size, n_embd)
        self.position_embedding_table = tf.keras.layers.Embedding(block_size, n_embd)
        self.blocks = tf.keras.models.Sequential(*[TransformerBlock() for _ in range(n_layer)])
        self.ln_f = tf.keras.layers.LayerNormalization(n_embd) # final layer norm
        self.lm_head = L.Linear(n_embd, vocab_size)

    def call(self, input):
        B, T = input.shape

        tok_emb = self.token_embedding_table(input)
        pos_emb = self.position_embedding_table(np.arange(T))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits =- self.lm_head(x)

        return logits

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = tf.nn.softmax(logits, axis=-1) # (B, C)
            # sample from the distribution
            #TODO: idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

            # append sampled index to the running sequence
            #TODO: idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
model = BigramLanguageModel()
model.compile(optimizer='adam', loss=tf.keras.losses.Crossentropy())
model.fit()

context = tf.ones((1,1), dtype=tf.float32)
print(decode(model.generate(context, max_new_tokens=2000)[0].to_list()))