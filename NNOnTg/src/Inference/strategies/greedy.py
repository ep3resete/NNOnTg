import tensorflow as tf
from .base import BaseStrategy


class GreedyStrategy(BaseStrategy):
    def generate_next_token(self, tokens, **kwargs):
        input_tensor = tf.constant([tokens], dtype=tf.int32)  # [1, seq_len]
        logits = self.model.call(input_tensor, training=False)  # [1, 10002] или [batch, 10002]
        
        # Универсальный способ - берем последний элемент батча
        if len(logits.shape) == 2:
            # [batch, vocab_size] - берем последний элемент батча
            last_batch_logits = logits[-1, :]  # [vocab_size]
        else:
            # На всякий случай другая форма
            last_batch_logits = logits[0, -1, :]
        
        next_token = tf.argmax(last_batch_logits).numpy()
        return next_token
