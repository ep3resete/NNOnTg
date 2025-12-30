import numpy as np
import json
import tensorflow as tf
from .transformer import TransformerBlock
from .layers import Embedding, PositionEncoder
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.layers import Input, Dense, Layer, Dropout
from keras import initializers


# training = False
# tf.config.run_functions_eagerly(True)


class GeneratorBlock(Layer):
    def __init__(self, vocab_size: int, dropout_rate):
        """ vocab_size - размер словаря в токенайзере. Нужно для последнего слоя"""
        self.dropout_rate = dropout_rate
        super().__init__()
        # self.dense1 = Dense(768, activation='relu')
        # self.dropout1 = Dropout(self.dropout_rate)
        # self.dense2 = Dense(512, activation='relu')
        # self.dropout2 = Dropout(self.dropout_rate)
        # self.dense3 = Dense(512, activation='relu')
        # self.dropout3 = Dropout(0.1)
        self.output_layer = Dense(
            vocab_size, 
            activation='softmax', 
            kernel_initializer=initializers.GlorotUniform(),
            bias_initializer=initializers.Zeros()
            )

    # def update_dropout_rate_generator(self, new_dropout_rate):
    #     self.dropout_rate = new_dropout_rate
    #     self.dropout1 = Dropout(self.dropout_rate)
    #     self.dropout2 = Dropout(self.dropout_rate)


    # @tf.function
    def call(self, inputs, training=False):
        """ Функция вызова. 
        inputs: tensor - входные эмбеддинги формы [batch_size, seq_lenght, d_model] """
        context_vector = inputs[:, -1, :] # Вектор контекста. (Последний эмбеддинг, содержит информацию обо всех предыдущих)
        # Прогон этого токена через полносвязные слои
        x = context_vector
        # x = self.dense1(context_vector) 
        # x = self.dropout1(x, training=training)
        # x = self.dense2(x)
        # x = self.dropout2(x, training=training)
        # x = self.dense3(x)
        # x = self.dropout3(x, training=training)
        x = self.output_layer(x)

        return x
    

class GPT(Model):

    def __init__(self, name_of_model, embedding_dim, seq_length, vocab_size, dropout_rate):
        """ 
        embedding_dim: int - размерность эмбеддингов
        seq_length: int - ФИКСИРОВАННАЯ длина входной последовательности
        vocab_size: int - размер словаря модели 
        """
        # with open("./config/PathConfig.json") as path_config_file:
        #     path_config = json.load(path_config_file)
        #     path_to_generator_config = path_config["GeneratorConfig"]
            
        # with open(path_to_generator_config) as generator_config_file:
        #     generator_config = json.load(generator_config_file)
        self.dropout_config = {'ffn_dropout': 0.35, 'residual_dropout': 0.2}
            

        super().__init__()
        self.name_of_model = name_of_model
        self.embedding_dim = embedding_dim
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.dropout_rate = dropout_rate
        self.inp = Input(shape=(None, ))
        self.embdding = Embedding(input_dim=vocab_size + 1, output_dim=embedding_dim, input_length=seq_length, embeddings_initializer=initializers.RandomNormal(mean=0.0, stddev=0.02),) 
        self.positional_encoder = PositionEncoder(d_model=embedding_dim)
        self.num_heads = 6
        self.transformer_block_1 = TransformerBlock(vocab_size=vocab_size, embedding_dim=embedding_dim, num_heads=self.num_heads, seq_length=seq_length, dropout_rate=dropout_rate)
        self.transformer_block_2 = TransformerBlock(vocab_size=vocab_size, embedding_dim=embedding_dim, num_heads=self.num_heads, seq_length=seq_length, dropout_rate=dropout_rate)
        self.transformer_block_3 = TransformerBlock(vocab_size=vocab_size, embedding_dim=embedding_dim, num_heads=self.num_heads, seq_length=seq_length, dropout_rate=dropout_rate)
        self.transformer_block_4 = TransformerBlock(vocab_size=vocab_size, embedding_dim=embedding_dim, num_heads=self.num_heads, seq_length=seq_length, dropout_rate=dropout_rate)
        # self.transformer_block_5 = TransformerBlock(vocab_size=vocab_size, embedding_dim=embedding_dim, num_heads=self.num_heads, seq_length=seq_length, dropout_rate=dropout_rate)
        # self.transformer_block_6 = TransformerBlock(vocab_size=vocab_size, embedding_dim=embedding_dim, num_heads=self.num_heads, seq_length=seq_length, dropout_rate=dropout_rate)
        # self.transformer_block_7 = TransformerBlock(vocab_size=vocab_size, embedding_dim=embedding_dim, num_heads=self.num_heads, seq_length=seq_length, dropout_rate=dropout_rate)
        # self.transformer_block_8 = TransformerBlock(vocab_size=vocab_size, embedding_dim=embedding_dim, num_heads=self.num_heads, seq_length=seq_length, dropout_rate=dropout_rate)
        # self.transformer_block_9 = TransformerBlock(vocab_size=vocab_size, embedding_dim=embedding_dim, num_heads=self.num_heads, seq_length=seq_length, dropout_rate=dropout_rate)
        # self.transformer_block_10 = TransformerBlock(vocab_size=vocab_size, embedding_dim=embedding_dim, num_heads=self.num_heads, seq_length=seq_length, dropout_rate=dropout_rate)
        # self.transformer_block_11 = TransformerBlock(vocab_size=vocab_size, embedding_dim=embedding_dim, num_heads=self.num_heads, seq_length=seq_length, dropout_rate=dropout_rate)
        # self.transformer_block_12 = TransformerBlock(vocab_size=vocab_size, embedding_dim=embedding_dim, num_heads=self.num_heads, seq_length=seq_length, dropout_rate=dropout_rate)

        self.generator_block = GeneratorBlock(vocab_size=vocab_size, dropout_rate=self.dropout_config["residual_dropout"])
    
    # @tf.function
    def call(self, inputs, training=False):
        """ На вход должны подаваться ТОКЕНЫ а не слова. 
        Так же в начало и конец сообщения нужно добавлять служебные символы начала и конца"""
        # Вход и позиционное кодирование эмбеддингов
        inputs = inputs[:, :self.seq_length] # Берутся только первые seq_length значения
        input_embeddings = self.embdding(inputs)
        position_encodered_embeddings = self.positional_encoder(input_embeddings)
        
        # Трансформерные блоки
        transformer_block_output1 = self.transformer_block_1(position_encodered_embeddings, training=training)
        transformer_block_output2 = self.transformer_block_2(transformer_block_output1, training=training)
        transformer_block_output3 = self.transformer_block_3(transformer_block_output2, training=training)
        transformer_block_output4 = self.transformer_block_4(transformer_block_output3, training=training)
        # transformer_block_output5 = self.transformer_block_5(transformer_block_output4, training=training)
        # transformer_block_output6 = self.transformer_block_6(transformer_block_output5, training=training)
        # transformer_block_output7 = self.transformer_block_7(transformer_block_output6, training=training)
        # transformer_block_output8 = self.transformer_block_8(transformer_block_output7, training=training)
        # transformer_block_output9 = self.transformer_block_9(transformer_block_output8, training=training)
        # transformer_block_output10 = self.transformer_block_10(transformer_block_output9, training=training)
        # transformer_block_output11 = self.transformer_block_11(transformer_block_output10, training=training)
        # transformer_block_output12 = self.transformer_block_12(transformer_block_output11, training=training)

        # Блок генератора
        generator_output = self.generator_block(transformer_block_output4, training=training)
        # generator_output = self.generator_block(transformer_block_output10, training=training)
        
        return generator_output
    
    def update_dropout_rate(self):
        current_dropout = 0.35
        for layer in self.layers:
            if hasattr(layer, 'dropout'):
                if isinstance(layer.dropout, float):
                    layer.dropout = current_dropout
                elif hasattr(layer, 'rate'):
                    layer.rate = current_dropout
            elif hasattr(layer, 'recurrent_dropout'):
                layer.recurrent_dropout = current_dropout
        
        # print(f"Epoch {epoch}: dropout_rate = {current_dropout:.3f}")

        # for layer in self.layers:
        #     if type(layer) == GeneratorBlock:
        #         layer.update_dropout_rate_generator(self.dropout_config["residual_dropout"])
        #     if type(layer) == TransformerBlock:
        #         layer.update_dropout_rate_ffn(self.dropout_config["ffn_dropout"])



    def get_config(self):
        """Сохраняет конфиг для сериализации"""
        config = super().get_config()
        config.update({
            'name_of_model': self.name_of_model,
            'embedding_dim': self.embedding_dim,
            'seq_length': self.seq_length, 
            'vocab_size': self.vocab_size,
            'dropout_rate': self.dropout_rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Создаёт экземпляр из конфига"""
        return cls(
            name_of_model=config['name_of_model'],
            embedding_dim=config['embedding_dim'],
            seq_length=config['seq_length'],
            vocab_size=config['vocab_size'],
            dropout_rate=config['dropout_rate']
        )
