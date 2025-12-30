import keras
import numpy as np
import tensorflow as tf
from collections import Counter

from Model import GPT
from Data import Dataset, BPETokenizer
import json

class LoadSetup:
    with open("./config/PathConfig.json", 'r', encoding='utf-8') as paths_file:
        paths_file = json.load(paths_file) # Все основные пути 
        path_to_input_config = paths_file["InputConfig"] # Путь к конфигу входа
        path_to_model = paths_file["PathToBestModel"]
        del paths_file # Удаление неиспользуемой переменной
        
    # Открыетие файла с конфигов для входа
    with open(path_to_input_config, 'r', encoding='utf-8') as config_input_file: 
        input_config = json.load(config_input_file) # Конфиг для входа
        seq_length = input_config["seq_length"] # Максимальная входная последовательность
        vocab_size = input_config["vocab_size"] + 2 # Размер слвоарного запаса модели (+2 так как токенизатор по итогу не учитывает некоторые служебные токены и создаёт vocab_size + 2 токенов)
        del input_config # Удаление лишней переменной


    tokenizer = BPETokenizer(learn=True)
    dataset = Dataset(False, None, False)
    train_token_counts = Counter(dataset.learneble_data[1])
    @classmethod
    @keras.saving.register_keras_serializable()
    def counter_loss(cls, y_true, y_pred):
        y_true_int = tf.cast(y_true, tf.int32)
        
        total = sum(cls.train_token_counts.values())
        weights = np.ones(cls.vocab_size, dtype=np.float32)
        
        for token_id, count in cls.train_token_counts.items():
            if 0 <= token_id < cls.vocab_size:
                freq = count / total
                # Более безопасная формула - например, обратный квадратный корень
                weights[token_id] = 1.0 / (freq**0.5 + 1e-8)
        
        # Ограничение максимального веса и нормализация
        weights = np.clip(weights, 1.0, 100.0)  # макс вес = 100
        weights = weights / np.mean(weights)
        weights = tf.constant(weights, dtype=tf.float32)
        
        loss = keras.losses.sparse_categorical_crossentropy(
            y_true_int, y_pred, from_logits=True
        )
        
        sample_weights = tf.gather(weights, y_true_int)
        
        return tf.reduce_mean(loss * sample_weights)

    # path_to_model = "./models/model_v_alpha_1.8.4.2.2__30.11.25_pre_train_fine_tuning_v2_256emd_4tr_6h1764530622.9390464.keras1764534581.3618088.keras"
    model: GPT = keras.models.load_model(path_to_model)
    model.name_of_model = "model_v_alpha_1.8.4.2.2__1.12.25_pre_train_fine_tuning_v2_256emd_4tr_6h"
