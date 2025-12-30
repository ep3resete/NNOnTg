from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.layers import Input, Embedding, Layer, MultiHeadAttention, LayerNormalization, Dropout
from .layers import FeedForwardNetwork
import tensorflow as tf
from keras import initializers

# Файл не менялся. Та самая лучшая модель обучалась с ним

class TransformerBlock(Layer):
    def __init__(self, vocab_size: int, embedding_dim: int, num_heads: int, seq_length: int, dropout_rate: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.seq_length = seq_length
        self.dropout_rate = dropout_rate
        
        self._causal_mask = self.get_causal_attention_mask(seq_length)
        
        # Всё для слоя MHA
        self.MHA_output_1 = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim, dropout=self.dropout_rate, kernel_initializer=initializers.GlorotUniform(),  bias_initializer=initializers.Zeros())
        self.MHA_norm_1 = LayerNormalization(epsilon=1e-8)

        # Всё для слоя FFN
        # self.FFN_output_1 = FeedForwardNetwork(hidden_dim=768, d_model=embedding_dim)
        self.FFN_output_1 = FeedForwardNetwork(hidden_dim=embedding_dim*4, d_model=embedding_dim)
        self.FFN_dropout = Dropout(self.dropout_rate)
        self.FFN_norm_1 = LayerNormalization(epsilon=1e-8)

    def update_dropout_rate_ffn(self, new_dropout_rate):
        self.dropout_rate = new_dropout_rate
        self.FFN_dropout = Dropout(self.dropout_rate)

    def get_causal_attention_mask(self, seq_len):
        mask = tf.linalg.band_part(
            tf.ones((seq_len, seq_len), dtype=tf.float32), -1, 0
        )
        
        return mask

    def call(self, inputs, training=False):
        """ Прямой проход через трансформерный блок. 
        inputs: tensor - входные эмбеддинги формы [batch_size, seq_lenght, d_model] 
        training: bool - Режим обучени (влияет на dropout)"""

        # Блок прогона через слой внимания. его нормализация и создание остаточного соединения
        mha_output_1 = self.MHA_output_1(
            query=inputs, 
            key=inputs, 
            value=inputs, 
            use_causal_mask=True,
            # attention_mask=causal_mask,
            training=training
            )
        mha_res_1 = mha_output_1 + inputs
        mha_norm_1 = self.MHA_norm_1(mha_res_1)

        # Блок прогона через FFN, его нормализация и создание остаточного соединения
        ffn_output_1 = self.FFN_output_1(mha_norm_1)
        ffn_dropout_1 = self.FFN_dropout(ffn_output_1)
        ffn_res_1 = ffn_dropout_1 + mha_norm_1
        ffn_norm_1 = self.FFN_norm_1(ffn_res_1)

        return ffn_norm_1
