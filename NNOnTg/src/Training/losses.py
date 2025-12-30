# import tensorflow as tf
# import numpy as np
# import keras
# from collections import OrderedDict, Counter


# # train_token_counts = Counter(dataset.learneble_data[1])

# class CounterLoss:
#     def __init__(self, train_token_counts, vocab_size):
#         # self.train_token_counts = train_token_counts
#         self.vocab_size = vocab_size
#         pass

#     @keras.saving.register_keras_serializable()
#     def counter_loss(y_true, y_pred, vocab_size=vocab_size):
#         y_true_int = tf.cast(y_true, tf.int32)
        
#         # Создаем массив весов
#         total = sum(train_token_counts.values())
#         weights = np.ones(vocab_size, dtype=np.float32)
        
#         for token_id, count in train_token_counts.items():
#             if 0 <= token_id < vocab_size:
#                 freq = count / total
#                 # Более безопасная формула - например, обратный квадратный корень
#                 weights[token_id] = 1.0 / (freq**0.5 + 1e-8)
        
#         # Ограничиваем максимальный вес и нормализуем
#         weights = np.clip(weights, 1.0, 100.0)  # макс вес = 100
#         weights = weights / np.mean(weights)
#         weights = tf.constant(weights, dtype=tf.float32)
        
#         # ПРАВИЛЬНЫЙ способ: взвешиваем саму функцию потерь
#         loss = keras.losses.sparse_categorical_crossentropy(
#             y_true_int, y_pred, from_logits=True
#         )
        
#         # Получаем веса для каждого примера в батче
#         sample_weights = tf.gather(weights, y_true_int)
        
#         return tf.reduce_mean(loss * sample_weights)
