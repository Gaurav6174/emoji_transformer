from typing import Optional

import tensorflow as tf

from config import DROPOUT_RATE
from model.attention import MultiHeadAttention


# feed-forward network block
class FeedForwardNetwork(tf.keras.layers.Layer):
    # two layer fully connected network
    def __init__(self, ff_dim: int, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

        self.dense_1 = tf.keras.layers.Dense(
            ff_dim, activation="relu", name="ffn_dense-1"
        )
        self.dense_2 = tf.keras.layers.Dense(embed_dim, name="ffn_dense_2")
        self.dropout = tf.keras.layers.Dropout(DROPOUT_RATE)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.dense_1(x)
        x = self.dropout(x, training=training)
        x = self.dense_2(x)
        x = self.dropout(x, training=training)
        return x

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "ff_dim": self.dense_1.units,
                "embed_dim": self.dense_2.units,
            }
        )
        return config


# encoder block
class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, **kwargs):
        self.dropout_1 = tf.keras.layers.Dropout(DROPOUT_RATE, name="dropout_1")
        self.dropout_2 = tf.keras.layers.Dropout(DROPOUT_RATE, name="dropout_2")
        
        super().__init__(**kwargs)
        self.supports_masking = True

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim

        # Sub-layer 1 — Multi-Head Self-Attention
        self.attention = MultiHeadAttention(
            embed_dim=embed_dim, num_heads=num_heads, name="multi_head_attention"
        )

        # Sub-layer 2 — Feed-Forward Network
        self.ffn = FeedForwardNetwork(
            ff_dim=ff_dim, embed_dim=embed_dim, name="feed_forward_network"
        )

        # Layer Norm 1 — applied after attention + residual
        # Layer Norm 2 — applied after FFN + residual
        self.layer_norm_1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name="layer_norm_1"
        )
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name="layer_norm_2"
        )

    def call(
        self, x: tf.Tensor, mask: Optional[tf.Tensor] = None, training: bool = False
    ) -> tf.Tensor:

        attention_out = self.attention(x, mask=mask, training=training)

        x = self.layer_norm_1(x + attention_out)
        x = self.dropout_1(x, training=training)
        
        ffn_out = self.ffn(x, training=training)

        x = self.layer_norm_2(x + ffn_out)
        x = self.dropout_2(x, training=training)
        return x

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
            }
        )
        return config
