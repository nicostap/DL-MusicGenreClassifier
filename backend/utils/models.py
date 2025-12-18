import tensorflow as tf
from tensorflow.keras import layers

class PositionalEmbedding(layers.Layer):
    def __init__(self, max_len=2048, d_model=128, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.pos_emb = layers.Embedding(input_dim=max_len, output_dim=d_model)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        return x + self.pos_emb(positions)

    def get_config(self):
        config = super().get_config()
        config.update({"max_len": self.pos_emb.input_dim, "d_model": self.d_model})
        return config