# multi-head self-attention implementation
from typing import Optional

import tensorflow as tf

from config import DROPOUT_RATE

# single attention head


class SingleAttentionHead(tf.keras.layers.Layer):
    def __init__(self, head_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.head_dim = head_dim
        # Q (Query)
        # K (Key)
        # V (Value)
        self.W_q = tf.keras.layers.Dense(head_dim, use_bias=False, name="W_q")
        self.W_k = tf.keras.layers.Dense(head_dim, use_bias=False, name="W_k")
        self.W_v = tf.keras.layers.Dense(head_dim, use_bias=False, name="W_v")

    def call(self, x: tf.Tensor, mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        # ff pass for one head
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # scaled dot-product
        scores = tf.matmul(Q, K, transpose_b=True)

        # scale sqrt(head_dim)
        scale = tf.constant(self.head_dim**0.5, dtype=tf.float32)
        scores = scores / scale

        # padding mask
        if mask is not None:
            mask = tf.squeeze(mask, axis=1)
            scores += (1.0 - tf.cast(mask, tf.float32)) * -1e9

        # softmax on last axis
        weights = tf.nn.softmax(scores, axis=-1)

        # weighted sum of vectors
        output = tf.matmul(weights, V)

        return output

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"head_dim": self.head_dim})
        return config


# multi head attention


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim: int, num_heads: int, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

        # validate split
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible bynum_heads ({num_heads})"
        )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.heads = [
            SingleAttentionHead(self.head_dim, name=f"head_{i}")
            for i in range(num_heads)
        ]

        # o/p projection
        self.W_o = tf.keras.layers.Dense(embed_dim, name="W_o")

        # adding dropout
        self.dropout = tf.keras.layers.Dropout(DROPOUT_RATE)

    def call(
        self, x: tf.Tensor, mask: Optional[tf.Tensor] = None, training: bool = False
    ) -> tf.Tensor:
        # ff pass multi_head, run all heads in parallel
        head_outputs = [head(x, mask=mask) for head in self.heads]

        concat = tf.concat(head_outputs, axis=-1)

        # linear projec + dropout
        output = self.W_o(concat)
        output = self.dropout(output, training=training)

        return output

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
            }
        )
        return config
