#Some code copying from NanoGPT.py
import math
import tensorflow as tf
from keras import layers

#Insert hyperparameters here
EPOCHS = 5
BATCH_SIZE = 64
ATTENT_SPAN = 64 #also known as our Time (T) component
N_EMBED = 64 #embeding count
N_HEAD = 8 #How many attention heads in a transformer
N_BLOCKS = 4 #Amount of transformers in our model
DROPOUT = 0.1
#

print("Starting NanoGPT attempt two")
print("\t{} Batch Size".format(BATCH_SIZE),
      "\t{} Time Slots (Attention Span)".format(ATTENT_SPAN),
      "\t{} Embeddings".format(N_EMBED),
      "\t{} Attention Heads".format(N_HEAD),
      "\t{} Transformers in Sequence".format(N_BLOCKS), sep="\n")

print("\nOpening Shakespeare file")
with open('shakespeare.txt', 'r', encoding='utf-8') as f:
	text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
print("\tCharacters: " + ''.join(chars))
print("\tVocab Size: {}".format(vocab_size))

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print("\nDefining Generator Class for Data")
data_raw = tf.constant(encode(text))
class PlaywrightGenerator(tf.keras.utils.Sequence):
    def __init__(self):
        super(PlaywrightGenerator, self).__init__()

    def __len__(self):
        return math.floor((len(data_raw) - (ATTENT_SPAN*2)) / BATCH_SIZE)

    def __getitem__(self, idx): #x is (B, T); y is (B) [so it's only guessing one token at a time]
        start = idx * BATCH_SIZE
        ix = tf.constant(range(start, start + BATCH_SIZE)) #we are going to basically "scan" through the text in batch size jumps/chunks

        x = tf.stack([data_raw[i:i+ATTENT_SPAN] for i in ix]) #(B, T) Batch of characters that are as long as the attention span we marked our system to have
        y = tf.constant(data_raw[start+ATTENT_SPAN:(start+ATTENT_SPAN+BATCH_SIZE)]) #(B) Batch of correct predictions to the next token
        return (x, y)
    
print("\nDefining Transformer Stuff")
class TransformerBlock(layers.Layer):
    def __init__(self):
        super(TransformerBlock, self).__init__()
        self.head_size = N_EMBED // N_HEAD
        

    def build(self, input_shape):
        self.sa = layers.MultiHeadAttention(num_heads=N_HEAD, key_dim=self.head_size, value_dim=self.head_size)

        self.ffwd = tf.keras.models.Sequential([
            layers.Dense(4 * N_EMBED),
            layers.Activation("relu"),
            layers.Dense(N_EMBED),
            layers.Dropout(DROPOUT),
        ])
        
        self.ln1 = layers.LayerNormalization()
        self.ln2 = layers.LayerNormalization()

    def call(self, x):
        n1 = self.ln1(x)
        x = x + self.sa(n1, n1, use_causal_mask=True) #Ensure only decoder
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(tf.keras.Model):
    def __init__(self):
        super(BigramLanguageModel, self).__init__()
        
    def build(self, input_shape):
        self.token_embedding_table = layers.Embedding(vocab_size, N_EMBED)
        self.position_embedding_table = layers.Embedding(ATTENT_SPAN, N_EMBED)

        self.blocks = tf.keras.models.Sequential([*[TransformerBlock() for _ in range(N_BLOCKS)]]) #Most of the magic
        self.ln_f = layers.LayerNormalization() # final layer norm
        self.lm_head = layers.Dense(vocab_size)
        self.single_char_converter = layers.Dense(vocab_size, activation='softmax')
    
    def call(self, input):
        T = tf.shape(input)[1]

        tok_emb = self.token_embedding_table(input) #(B, T, C)
        pos_emb = self.position_embedding_table(tf.range(T)) #(T, C)
        x = tok_emb + pos_emb #(B, T, C) + (T, C) = (B, T, C) bc it will apply (T, C) to all of B
        x = self.blocks(x) #(B, T, C)
        x = self.ln_f(x) #(B, T, C)
        logits = self.lm_head(x) #(B, T, vocab_size)
        #if we were to stop here, this would produce the next sequence of tokens it thinks comes afterwards (in order)
        #but we want it to simply guess the next character in the sequence, we don't care about the next sentence, for now
        logits = layers.Flatten()(logits) #(B, T*vocab_size)
        logits = self.single_char_converter(logits) #(B, vocab_size), in probabilities

        return logits
        
    def generate(self, idx, max_new_tokens=500):
        #expecting idx to be in shape (1, T)
        for _ in range(max_new_tokens):
            idx_time = idx[:, -1*ATTENT_SPAN:] # Get the last time sequence
            probs = self(idx_time) #Get the most probable next character
            next_char = tf.math.argmax(probs, 1) #since probs is a tensor of probabilities, the one with the highest probability will be given

            next_char = tf.expand_dims(next_char, axis=0)
            #And then concatenate that character to the previously generated stuff, and repeat
            idx = tf.raw_ops.Concat(concat_dim=1, values=(idx, tf.cast(next_char, dtype=tf.float32)))
        
        return idx

print("\nTraining")
model = BigramLanguageModel()
model.compile("adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
model.fit(PlaywrightGenerator(), epochs=EPOCHS)

print("\nGenerating")
context = tf.ones((1,ATTENT_SPAN-1), dtype=tf.float32)
context = tf.raw_ops.Concat(concat_dim=1, values=(context, tf.constant([[0]], dtype=tf.float32)))
print(decode(model.generate(context, max_new_tokens=500)[0].numpy()))