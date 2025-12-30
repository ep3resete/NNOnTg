import tensorflow as tf
from .base import BaseStrategy



class TopPStrategy(BaseStrategy):
    def __init__(self, model, tokenizer, top_p=0.1, temperature=0.4):
        super().__init__(model, tokenizer)
        self.top_p = top_p
        self.temperature = temperature

    def generate_next_token(self, tokens, **kwargs):
        """
        Метод для определения следующего токена.
    
        Аргументы:        
            tokens - Токены промпта, который подаётся на вход модели.
        """
        if 'top_p' in kwargs:
            self.top_p = kwargs.get('top_p', self.top_p)
        if 'temperature' in kwargs:
            self.temperature = kwargs.get('temperature', self.top_p)

        input_tensor = tf.constant([tokens], dtype=tf.int32)
        logits = self.model.call(input_tensor, training=False)
        indices = tf.constant([[0, 1]])  # [batch_index, token_index]
        updates = tf.constant([1e-15])

        logits = tf.tensor_scatter_nd_update(logits, indices, updates)

        filtered_logits: tf.Tensor = self.top_p_filter(logits, self.top_p)
        
        scaled_logits = filtered_logits / self.temperature
        probabilities = tf.nn.softmax(scaled_logits).numpy()
        next_token = tf.random.categorical(probabilities, 1).numpy()

        return next_token
    
    def top_p_filter(self, logits: tf.Tensor, p: float):
        """Метод для фильтрации логитов по методу top P"""
        batch_size = tf.shape(logits)[0]
        filtered_logits_list = []
        
        for i in range(batch_size):
            single_logits = logits[i]
            
            # Получение вероятности из логитов
            probabilities = single_logits
            # probabilities = tf.nn.softmax(single_logits, axis=-1)
            
            # Сортировка вероятности по убыванию
            sorted_probs = tf.sort(probabilities, direction='DESCENDING', axis=-1)
            sorted_indices = tf.argsort(probabilities, direction='DESCENDING', axis=-1)
            
            # Вычисление кумулятивной суммы вероятностей
            cumulative_probs = tf.cumsum(sorted_probs, axis=-1)
            
            # Создание маски для токенов, которые нужно СОХРАНИТЬ
            keep_mask = cumulative_probs <= p
            
            # Если все False, остаётся первый самый вероятный
            depth = tf.shape(keep_mask)[-1]
            first_token_mask = tf.one_hot(0, depth=depth, on_value=True, off_value=False)
            keep_mask = tf.logical_or(keep_mask, first_token_mask)
            
            # Шаг 6: Восстановление маски в исходном порядке токенов
            original_order_keep_mask = tf.gather(keep_mask, sorted_indices, batch_dims=0)  
            
            # Шаг 7: Применение маски к исходным логитам
            filtered_single_logits = tf.where(
                original_order_keep_mask,
                single_logits,
                tf.constant(-1e10, dtype=single_logits.dtype)
            )
            
            filtered_logits_list.append(filtered_single_logits)
        
        return tf.stack(filtered_logits_list, axis=0)
