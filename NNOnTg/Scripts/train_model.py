import os
import sys
import pickle
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, 'src')

# Удаление всех существующих путей и добавление пути с src
sys.path = [src_path] + [p for p in sys.path if p != src_path]


from keras._tf_keras.keras.optimizers import Adam, AdamW, RMSprop, Adafactor
from keras._tf_keras.keras.losses import SparseCategoricalCrossentropy
from keras._tf_keras.keras.metrics import SparseCategoricalAccuracy, SparseTopKCategoricalAccuracy
import keras
import numpy as np
import tensorflow as tf
from collections import OrderedDict, Counter



from Model import GPT
from Training import GPTTrainer
from Data import Dataset, BPETokenizer
from Utils import LoadSetup
from load_tokenizer import load_tokenizer
import predict
# from src.Model import GPT
# from src.Training import GPTTrainer
# from src.Data import Dataset, BPETokenizer
# from src.Utils import LoadSetup

# model = GPT('model_test', 512, 10, 100)


# vocab_size1 = 21069
# vocab_size = 10002
# seq_lenght = 128

# # import tensorflow as tf


def main():
    ls = LoadSetup()
    counter_loss = ls.counter_loss
    model = ls.model
    dataset = ls.dataset
    tokenizer = ls.tokenizer
    model.get_config()
    model.summary()
    trainer = GPTTrainer(model, dataset, tokenizer, 0.1)
    # print(trainer.fit())
    predict.main(model)

if __name__ == '__main__':
    main()

# def main():
#     tokenizer = BPETokenizer(learn=True)
#     dataset = Dataset(False, None, False)
#     train_token_counts = Counter(dataset.learneble_data[1])


#     @keras.saving.register_keras_serializable()
#     def counter_loss(y_true, y_pred):
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

#     # def counter_loss(y_true, y_pred):
#     #     y_true_int = tf.cast(y_true, tf.int32)
        
#     #     # Берем статистику из Counter и создаем веса
#     #     # counts - это твой Counter с частотами токенов
#     #     counts = train_token_counts  # твой Counter объект
        
#     #     # Создаем массив весов на основе частот
#     #     total = sum(counts.values())
#     #     weights = np.ones(vocab_size, dtype=np.float32)
        
#     #     for token_id, count in counts.items():
#     #         if 0 <= token_id < vocab_size:
#     #             freq = count / total
#     #             # Инвертируем частоты: редкие токены получают больший вес
#     #             weights[token_id] = 1.0 / (freq + 1e-8)
        
#     #     # Нормализуем веса
#     #     weights = weights / np.mean(weights)
#     #     weights = tf.constant(weights, dtype=tf.float32)
        
#     #     # Применяем веса к логитам перед softmax
#     #     weighted_logits = y_pred * weights
        
#     #     # Стандартная кросс-энтропия с взвешенными логитами
#     #     loss = keras.losses.sparse_categorical_crossentropy(
#     #         y_true_int, weighted_logits, from_logits=True
#     #     )
        
#     #     return tf.reduce_mean(loss)

#     # from sklearn.utils.class_weight import compute_class_weight
#     # # 1. Функция расчета весов
#     # def calculate_class_weights_sklearn(token_counts, vocab_size):
#     #     # Собираем все метки в один массив
#     #     all_labels = []
#     #     for token_id, count in token_counts.items():
#     #         if 0 <= token_id < vocab_size:
#     #             all_labels.extend([token_id] * count)
        
#     #     all_labels = np.array(all_labels)
        
#     #     # Вычисляем веса по проверенной формуле
#     #     weights = compute_class_weight(
#     #         'balanced',
#     #         classes=np.arange(vocab_size),
#     #         y=all_labels
#     #     )
    
#     #     print(f"Sklearn веса: min={np.min(weights):.3f}, max={np.max(weights):.3f}")
#     #     return tf.constant(weights, dtype=tf.float32)
    
#     # # Функция потерь
#     # def weighted_sparse_categorical_crossentropy(y_true, y_pred):
#     #     y_true_int = tf.cast(y_true, tf.int32)
#     #     loss = keras.losses.sparse_categorical_crossentropy(y_true_int, y_pred, from_logits=True)
#     #     gathered_weights = tf.gather(class_weights, y_true_int)
#     #     weighted_loss = loss * gathered_weights
#     #     return tf.reduce_mean(weighted_loss)

#     # Использование
#     # model: GPT = keras.models.load_model("./models/model_v_alpha_1.8.4.2_pre_train_fine_tuning_256emd_4tr_6h1763894298.8380203.keras1763894437.3605213.keras")
#     # model: GPT = keras.models.load_model("./models/model_v_alpha_1.8.4.2_pre_train_fine_tuning_256emd_4tr_6h1763886968.1354656.keras1763890271.9500487.keras")
#     # model: GPT = keras.models.load_model("./models/model_v_alpha_1.8.4.3_pre_train_256emd_4tr_6h1764168586.9873629.keras1764172002.7981217.keras")
#     model: GPT = keras.models.load_model("./models/model_v_alpha_1.8.4.2.2__30.11.25_pre_train_fine_tuning_v2_256emd_4tr_6h1764530622.9390464.keras1764534581.3618088.keras")
#     model.name_of_model = "model_v_alpha_1.8.4.2.2__1.12.25_pre_train_fine_tuning_v2_256emd_4tr_6h"
#     # model = GPT('model_v_alpha_1.8.4.3_pre_train_256emd_4tr_6h', 256, seq_lenght, vocab_size, 0.3)
#     # lr_schedule = keras.optimizers.schedules.CosineDecay(
#     # 5e-4, 100000  # 100k шагов
#     # )
#     # optimizer = AdamW(
#     # learning_rate=1e-4,  
#     # epsilon=1e-8,
#     # beta_1=0.9,
#     # beta_2=0.98,
#     # weight_decay=0.01,
#     # global_clipnorm=1.0
#     # )

#     # model.compile(
#     #     optimizer=optimizer,
#     #     # loss=custom_loss_2, 
#     #     loss=counter_loss, 
#     #     # loss=weighted_sparse_categorical_crossentropy, 
#     #     metrics=[ 
#     #         SparseCategoricalAccuracy(name='accuracy'), 
#     #         SparseTopKCategoricalAccuracy(k=5, name='top5_accuracy'), 
#     #         SparseTopKCategoricalAccuracy(k=10, name='top10_accuracy'),
#     #         ],
#     #     jit_compile=True
#     #     )
#     #         # 'accuracy', 'perplexity', 
#     model.get_config()
#     model.summary()
#     trainer = GPTTrainer(model, dataset, tokenizer, 0.1)
#     print(trainer.fit())
#     predict.main(model)


# if __name__ == '__main__':
#     main()

# # import pickle

# # with open('./data/tokenizers/tokenizer_saved.pkl', 'wb') as f:
# #     pickle.dump(tokenizer, f)
# # [[3, 11, 5, 12, 69, 4, 2], [11, 6, 22, 34, 5, 1007, 296, 1231, 7, 19, 37, 48, 4, 2], [3, 64, 5, 60, 5, 17, 681, 18, 268, 7, 2], [128, 4542, 1439, 9, 3946, 977, 5, 1497, 226, 10, 116, 7, 2]]
