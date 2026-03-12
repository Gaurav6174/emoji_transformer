import tensorflow as tf

from config import (
    DROPOUT_RATE,
    EMBED_DIM,
    FF_DIM,
    MAX_SEQ_LEN,
    NUM_BLOCKS,
    NUM_CLASSES,
    NUM_HEADS,
    VOCAB_SIZE,
)
from model.encoder_block import EncoderBlock
from model.positional_encoding import PositionalEmbedding


class Transformer(tf.keras.Model):
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        max_seq_len: int = MAX_SEQ_LEN,
        embed_dim: int = EMBED_DIM,
        num_heads: int = NUM_HEADS,
        ff_dim: int = FF_DIM,
        num_blocks: int = NUM_BLOCKS,
        num_classes: int = NUM_CLASSES,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Store for get_config()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_blocks = num_blocks
        self.num_classes = num_classes

        # input -> Embedding
        self.embedding = PositionalEmbedding(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            max_seq_len=max_seq_len,
            name="positional_embedding",
        )

        # stack of encoder_blocks
        self.encoder_blocks = [
            EncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                name=f"encoder_block_{i}",
            )
            for i in range(num_blocks)
        ]

        # sequence -> single vector
        self.pooling = tf.keras.layers.GlobalAveragePooling1D(name="global_avg_pool")

        # classification head
        self.dropout_1 = tf.keras.layers.Dropout(DROPOUT_RATE, name="dropout_1")
        self.dropout_2 = tf.keras.layers.Dropout(DROPOUT_RATE, name="dropout_2")

        # intermediate dense layer
        self.dense = tf.keras.layers.Dense(
            ff_dim, activation="relu", name="classifier_dense"
        )

        # output layer
        self.output_layer = tf.keras.layers.Dense(
            num_classes, activation="softmax", name="classifier_output"
        )

    # forward pass
    def call(
        self,
        inputs: tf.Tensor,
        training: bool | None = None,
        mask=None,
    ) -> tf.Tensor:
        del mask
        train_flag = bool(training) if training is not None else False

        # padding mask
        attn_mask = self._make_padding_mask(inputs)
        seq_mask = tf.cast(tf.math.not_equal(inputs, 0), dtype=tf.float32)

        x = self.embedding(inputs)

        for block in self.encoder_blocks:
            x = block(x, mask=attn_mask, training=train_flag)

        x = self.pooling(x, mask=seq_mask)

        # classification head
        x = self.dropout_1(x, training=train_flag)
        x = self.dense(x)
        x = self.dropout_2(x, training=train_flag)
        x = self.output_layer(x)

        return x

    # padding mask

    def _make_padding_mask(self, x: tf.Tensor) -> tf.Tensor:

        mask = tf.cast(tf.math.not_equal(x, 0), dtype=tf.float32)

        mask = mask[:, tf.newaxis, tf.newaxis, :]

        return mask

    # model summary
    def build_and_summarise(self) -> None:
        dummy_input = tf.zeros((1, MAX_SEQ_LEN), dtype=tf.int32)
        self(dummy_input, training=False)
        self.summary()

    # serialisation
    def get_config(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "num_blocks": self.num_blocks,
            "num_classes": self.num_classes,
        }

    @classmethod
    def from_config(
        cls, config: dict, custom_objects: dict | None = None
    ) -> "Transformer":
        del custom_objects
        return cls(**config)
