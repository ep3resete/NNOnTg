import tensorflow as tf
from .base import BaseStrategy

class SamplingStrategy(BaseStrategy):
    def __init__(self, model, tokenizer, temperature=1.0):
        """ Класс для простого сэмплирования """
        super().__init__(model, tokenizer)
        self.temperature = temperature
    
    def generate_next_token(self, tokens, **kwargs):
        if 'temperature' in kwargs:
            self.temperature = kwargs.get('temperature', self.top_p)

        input_tensor = tf.constant([tokens]) # Преобразование входных данных в тензор
        logits = self.model.call(input_tensor, training=False) # Получение логитов 
        indices = tf.constant([[0, 1]])  # [batch_index, token_index]
        updates = tf.constant([1e-15])
        
        # Обновление логитов
        logits = tf.tensor_scatter_nd_update(logits, indices, updates)

        
        scaled_logits = logits / self.temperature # Сэмплирование по температуре
        probabilities = tf.nn.softmax(scaled_logits).numpy() # Софтмакс преобразование
        next_token = tf.random.categorical(probabilities, 1).numpy() # Следующий токен

        return next_token