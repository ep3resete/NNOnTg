import tensorflow as tf
import numpy as np
import json
from keras._tf_keras.keras.layers import Embedding, Layer, Dense, Dropout, LayerNormalization
from keras import initializers

with open("config/PathConfig.json", 'r', encoding='utf-8') as path_config_file:
    path_to_input_file_config = json.load(path_config_file)["InputConfig"]

with open(path_to_input_file_config, 'r', encoding='utf-8') as input_settings_file:
    input_settings = json.load(input_settings_file)



vocab_size = input_settings["vocab_size"]
embedding_dim = input_settings["embedding_dim"]
seq_lenght = input_settings["seq_length"]

class PositionEncoder(Layer):
    """ Позиционное кодирование эмбеддинга """
    def __init__(self, d_model, max_len=5000, **kwargs):
        super(PositionEncoder, self).__init__(**kwargs)
        self.d_model = d_model
        self.max_len = max_len
        self.pos_encoding = self.create_matrix_of_pos_encoding(d_model, max_len)

    def create_matrix_of_pos_encoding(self, d_model, max_length):
        """ d_model - размер каждого эмбеддинга
        max_length - максимальная длина входных данных
        """
        position = tf.range(max_length, dtype=tf.float32)[:, tf.newaxis]  # [max_length, 1]
        div_term = tf.exp(
                    tf.range(0, d_model, 2, dtype=tf.float32) *  # [d_model//2]
                    (-tf.math.log(10000.0) / d_model)  # скаляр
                )  # [d_model//2]
        sin_enc = tf.sin(position * div_term)
        cos_enc = tf.cos(position * div_term)
        pos_encoding = tf.stack([sin_enc, cos_enc], axis=-1)
        self.pos_encoding = tf.reshape(pos_encoding, [max_length, d_model]) # Матрица для кодирования. Строки - векторы позиционного кодирования для каждого элемента из последовательности
        
        return self.pos_encoding

    def call(self, inputs):
        seq_length = tf.shape(inputs)[1]
        return inputs + self.pos_encoding[:seq_length, :]


class FeedForwardNetwork(Layer):
    """ Класс для FFN принимает на вход матрицу, полученную в результате блока внимания MHA """
    def __init__(self, d_model, hidden_dim, dropout_rate=0.1, activation='gelu', **kwargs):
        super().__init__(**kwargs)
        """ d_model - размер каждого эмбеддинга 
        hidden_dim - размер скрытых слоёв в FFN
        dropout_rate - коэффицент дропаута 
        activation - функция активации """
        self.d_model = d_model
        
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.activation = activation

        self.dense1 = Dense(
            hidden_dim, 
            activation=activation,
        
            kernel_initializer='he_normal',
            bias_initializer=initializers.Zeros()
            ) # Первый скрытый слой (Расширение)
        
        self.dropout1 = Dropout(dropout_rate) # Дропаут 
        self.dense1_norm_1 = LayerNormalization(epsilon=1e-8)
        
        self.dense2 = Dense(
            d_model, 
            kernel_initializer=initializers.GlorotUniform(),
            bias_initializer=initializers.Zeros()
            ) # Второй скрытый слой (Возврат к исходномук размеру)
        
        self.dropout2 = Dropout(dropout_rate) # Дропаут
        self.dense2_norm_1 = LayerNormalization(epsilon=1e-8)


    def call(self, inputs, training=False):
        # return super().call(*args, **kwargs)
        x = self.dense1(inputs)
        x = self.dropout1(x, training=training)
        x = self.dense1_norm_1(x)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        x = self.dense2_norm_1(x)

        return x + inputs

    def get_config(self):
        config = super().get_config()
        config.update({'d_model': self.d_model, 
                       'hidden_dim': self.hidden_dim, 
                       'dropout_rate': self.dropout_rate, 
                       'activation': self.activation})
        return config
