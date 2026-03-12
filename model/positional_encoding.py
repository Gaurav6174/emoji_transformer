import numpy as np
import tensorflow as tf
from config import MAX_SEQ_LEN, EMBED_DIM

#ENCODING MATRIX

def get_positional_encoding(seq_len: int, embed_dim: int) -> tf.Tensor:
    #using fixed sine/cosine formula
    
    positions = np.arange(seq_len).reshape(-1, 1)
    dims = np.arange(embed_dim // 2).reshape(1, -1)
    angles = positions / np.power(10_000, (2*dims) / embed_dim)
    
    encoding = np.zeros((seq_len, embed_dim))
    encoding[:, 0::2] = np.sin(angles)
    encoding[:, 1::2] = np.cos(angles)

    encoding = encoding[np.newaxis, :, :]
    
    return tf.cast(encoding, dtype = tf.float32)

#KERAS LAYER WRAPPER
class PositionalEmbedding(tf.keras.layers.Layer):
    #1.Token Embedding
    #2. Postional Encoding
    
    def __init__(self, vocab_size: int, embed_dim: int, max_seq_len: int, **kwargs):
        super().__init__(**kwargs)

        self.vocab_size  = vocab_size
        self.embed_dim   = embed_dim
        self.max_seq_len = max_seq_len
        
        #Learnable lookup, these weights are updated by backprop
        self.token_embedding = tf.keras.layers.Embedding(
            input_dim = vocab_size,
            output_dim = embed_dim,
            #mask_zero = True,
            name = "token_embedding"
        )
        
        self.pos_encoding = get_positional_encoding(max_seq_len, embed_dim)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        #Forward Pass
        seq_len = tf.shape(x)[1]
        
        x = self.token_embedding(x)
        
        x *= tf.constant(self.embed_dim ** 0.5, dtype=tf.float32)

        x += self.pos_encoding[:, :seq_len, :]
        
        return x

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "max_seq_len": self.max_seq_len,
        })
        return config

            

    
    
    