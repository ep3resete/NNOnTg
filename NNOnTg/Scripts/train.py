
#     dataset = Dataset()
#     # model = keras._tf_keras.keras.models.load_model("./models/model_v_alpha_1.3.11761982137.2486532.keras1762001272.5561724.keras")
#     # model: GPT = keras._tf_keras.keras.models.load_model("./models/model_v_alpha_1.3.11761982137.2486532.keras1762001919.1295054.keras")
#     # model: GPT = keras._tf_keras.keras.models.load_model("./models/model_v_alpha_1.3.11761982137.2486532.keras1761988727.689457.keras")
#     # model.name_of_model + 'rem_v2'
#     # model.name_of_model += "rem_v1"
#     # model = keras._tf_keras.keras.models.load_model("./models/model_v_alpha_1.3.11762022383.7852242.keras1762029729.5242062.keras")
#     model = GPT('model_v_alpha_1.5.1_word', 512, seq_lenght, vocab_size, 0.2)
#     # model = GPT('model_v_alpha_1.4.1.2', 512, seq_lenght, vocab_size, 0.2)
#     # model = GPT('model_v_alpha_1.4.1.2', 512, seq_lenght, vocab_size, 0.2)
#     # model.load_weights("./models/model_weights_40_epochs.weights.h5")
#     # optimizer = Adam(learning_rate=0.00001, clipnorm=1.0)  # вместо 1e-3 или больше
#     optimizer = AdamW(


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
from load_tokenizer import load_tokenizer
import predict


vocab_size1 = 21069
vocab_size = 30002
seq_lenght = 128

import tensorflow as tf


# Установка seed
# tf.random.set_seed(42)

@tf.function(jit_compile=True)
def custom_loss(y_true, y_pred):
    # Используем from_logits=True если ваша модель не использует softmax на выходе
    base_loss = keras.losses.sparse_categorical_crossentropy(
        y_true, y_pred, from_logits=True
    )
    
    repetition_penalty = tf.constant(0.001, dtype=tf.float32)
    
    # Оптимизированный расчет с избежанием создания больших one-hot матриц
    y_true_flat = tf.reshape(y_true, [-1])  # flatten для эффективности
    vocab_size = tf.shape(y_pred)[-1]
    
    # Быстрая гистограмма
    token_counts = tf.math.unsorted_segment_sum(
        tf.ones(tf.shape(y_true_flat)[0], dtype=tf.float32),
        tf.cast(y_true_flat, tf.int32),
        num_segments=vocab_size
    )
    
    penalty = repetition_penalty * tf.reduce_sum(token_counts ** 2)
    
    return base_loss + penalty


@tf.function(jit_compile=True)
def custom_loss_2(y_true, y_pred):
    vocab_size = tf.shape(y_pred)[-1]
    
    # Создание one-hot encoding таргетов
    one_hot_targets = tf.one_hot(tf.cast(y_true, tf.int32), vocab_size)
    
    # Применение label smoothing
    smoothing = 0.01
    smoothed_targets = one_hot_targets * (1.0 - smoothing) + smoothing / tf.cast(vocab_size, tf.float32)
    
    # Вычисление cross-entropy
    loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=smoothed_targets, 
        logits=y_pred
    )
    
    return loss


@tf.function(jit_compile=True)
class FocalLoss(keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.0, vocab_size=9002):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.vocab_size = vocab_size

    def call(self, y_true, y_pred):
        # y_true: [batch_size], y_pred: [batch_size, vocab_size]
        
        #  y_true -> one-hot для получения вероятностей
        y_true_one_hot = tf.one_hot(y_true, depth=self.vocab_size)
        
        # Вычисление вероятности предсказаний
        probabilities = tf.nn.softmax(y_pred, axis=-1)
        
        # Выбор вероятности правильных классов
        p_t = tf.reduce_sum(y_true_one_hot * probabilities, axis=-1)
        
        # Вычисление cross entropy
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        
        # Focal loss компонент
        focal_loss = self.alpha * tf.pow(1 - p_t, self.gamma) * ce
        
        return tf.reduce_mean(focal_loss)


def frequency_penalized_loss(y_true, y_pred, token_frequencies):
    """
    Лосс который штрафует предсказание частых токенов
    """
    penalty_strength=2.0
    base_loss = keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    
    # Получение предсказания модели
    predictions = tf.argmax(y_pred, axis=-1)
    
    # Создание маски частых токенов
    frequent_tokens_mask = tf.cast(
        tf.gather(token_frequencies, tf.cast(predictions, tf.int32)) > 0.1, 
        tf.float32
    )
    
    # Штраф частых предсказаний
    penalty = frequent_tokens_mask * penalty_strength
    
    return base_loss + penalty

def main():
    tokenizer = BPETokenizer(learn=True)
    # tokenizer.load_tokenizer("./data/tokenizers/bpe_tokenizer1.json")
    # tokenizer.load_tokenizer("./data/tokenizers/bpe_tokenizer.json")

    dataset = Dataset(False, None, False)
    from sklearn.utils.class_weight import compute_class_weight
    # 1. Функция расчета весов
    def calculate_class_weights_sklearn(token_counts, vocab_size):
        # Собираем все метки в один массив
        all_labels = []
        for token_id, count in token_counts.items():
            if 0 <= token_id < vocab_size:
                all_labels.extend([token_id] * count)
        
        all_labels = np.array(all_labels)
        
        # Вычисляем веса по проверенной формуле
        weights = compute_class_weight(
            'balanced',
            classes=np.arange(vocab_size),
            y=all_labels
        )
    
        print(f"Sklearn веса: min={np.min(weights):.3f}, max={np.max(weights):.3f}")
        return tf.constant(weights, dtype=tf.float32)
    
    # Функция потерь
    def weighted_sparse_categorical_crossentropy(y_true, y_pred):
        y_true_int = tf.cast(y_true, tf.int32)
        loss = keras.losses.sparse_categorical_crossentropy(y_true_int, y_pred, from_logits=True)
        gathered_weights = tf.gather(class_weights, y_true_int)
        weighted_loss = loss * gathered_weights
        return tf.reduce_mean(weighted_loss)

    # Использование
    train_token_counts = Counter(dataset.learneble_data[1])
    class_weights = calculate_class_weights_sklearn(train_token_counts, vocab_size)

    # Компиляция модели
    # model.compile(
    #     optimizer='adam',
    #     loss=weighted_sparse_categorical_crossentropy,
    #     metrics=['accuracy']
    # )


    # dataset = Dataset(True, tokenizer.index_docs, True)
    # model: GPT = keras._tf_keras.keras.models.load_model("./models/model_v_alpha_1.3.1rem_v2.11762186816.4859521.keras1762189002.6943734.keras")

    # model = keras._tf_keras.keras.models.load_model("./models/model_v_alpha_1.3.11761982137.2486532.keras1762001272.5561724.keras")
    # model: GPT = keras._tf_keras.keras.models.load_model("./models/model_v_alpha_1.3.1rem_v2.11762186816.4859521.keras1762189002.6943734.keras")
    # model: GPT = keras._tf_keras.keras.models.load_model("./models/model_v_alpha_1.3.1rem_v2.11762186816.4859521.keras1762189002.6943734.keras")
    # model: GPT = keras._tf_keras.keras.models.load_model("./models/model_v_alpha_1.3.1rem_v2.11762183046.6873677.keras1762186145.300428.keras")
    # model: GPT = keras._tf_keras.keras.models.load_model("./models/model_v_alpha_1.3.1_skipped_fequerentrem_v2.11762243624.9922683.keras1762255467.1056027.keras")
    # model: GPT = keras._tf_keras.keras.models.load_model("./models/model_v_alpha_1.3.1_skipped_fequerentrem_v2.11762243624.9922683.keras1762255467.1056027.keras")
    # model: GPT = keras._tf_keras.keras.models.load_model("./models/model_v_alpha_1.3.1_skipped_fequerentrem_v2.11762243624.9922683.keras1762255467.1056027.keras")


    # model: GPT = keras._tf_keras.keras.models.load_model("./models/model_v_alpha_1.3.1_skipped_fequerentrem_v2.11762243624.9922683.keras1762255467.1056027.keras") # Нормально обучаемая сеть 1.3.1 3 эпоха, хорошие веса сгенериоись, можно использовать для тестов с обчуением
    # model: GPT = keras._tf_keras.keras.models.load_model("./models/model_v_alpha_1.3.1_skipped_fequerentrem_v2.11762456009.3512404.keras1762460106.4230816.keras")
    
    # model: GPT = keras._tf_keras.keras.models.load_model("./models/model_v_alpha_1.3.1_skipped_fequerentrem_v2.11762456009.3512404.keras1762460555.3820436.keras") # Нормально обучаемая сеть 1.3.1 3 эпоха, хорошие веса сгенериоись, можно использовать для тестов с обчуением
    # model: GPT = keras._tf_keras.keras.models.load_model("./models/model_v_alpha_1.3.1_skipped_fequerentrem_v2.11762243624.9922683.keras1762255467.1056027.keras") # Нормально обучаемая сеть 1.3.1 3 эпоха, хорошие веса сгенериоись, можно использовать для тестов с обчуением
    # model.name
    # ПОСЛЕДНЯЯ МОДЕЛЬ 0:22 7.11.2025
    # ./models/model_v_alpha_1.3.1_skipped_fequerentrem_v2.11762456009.3512404.keras1762460106.4230816.keras <--- ПОСЛЕДНЯЯ МОДЕЛЬ 0:22 7.11.2025 Норм резы по логам
    # ./models/model_v_alpha_1.3.1_skipped_fequerentrem_v2.11762456009.3512404.keras1762460106.4230816.keras <--- ПОСЛЕДНЯЯ МОДЕЛЬ 0:22 7.11.2025 Норм резы по логам
    # "./models/model_v_alpha_1.3.1_skipped_fequerentrem_v2.11762456009.3512404.keras1762460555.3820436.keras"
    # ПОСЛЕДНЯЯ МОДЕЛЬ 0:22 7.11.2025
    # model.name_of_model = "model_v_alpha_1.6.1_word"
    # model.name_of_model = "model_v_alpha_1.7_word_custom_loss"


    
    # model: GPT = keras._tf_keras.keras.models.load_model("./models/model_v_alpha_1.3.11762201613.2739942.keras1762203656.4552946.keras")
    # model: GPT = keras._tf_keras.keras.models.load_model("./models/model_v_alpha_1.3.1rem_v2.11762175482.564715.keras1762176477.7573154.keras")
    # model: GPT = keras._tf_keras.keras.models.load_model("./models/model_v_alpha_1.6.1.4_re5_dataset1762670655.2237139.keras1762678963.9153116.keras")
    # model: GPT = keras.models.load_model("./models/model_v_alpha_1.8.3_pre_train_fine_tuning_256emd_4tr_6h1763565733.3929126.keras1763567714.8885152.keras")
    # model: GPT = keras.models.load_model("./models/model_v_alpha_1.8.3_pre_train_fine_tuning_256emd_4tr_6h1763565733.3929126.keras1763567714.8885152.keras")
    # weights = model.get_weights()
    # model = GPT('model_v_alpha_1.8.3_pre_train_fine_tuning_256emd_4tr_6h', 256, seq_lenght, vocab_size, 0.4)
    # model.set_weights(weights)
    # model: GPT = keras.models.load_model("./models/model_v_alpha_1.8.3_pre_train_fine_tuning_256emd_4tr_6h1763562750.6335526.keras1763564362.111384.keras")
    # model: GPT = keras.models.load_model("./models/model_v_alpha_1.8.3_pre_train_fine_tuning_256emd_4tr_6h1763552503.6560915.keras1763562605.7316828.keras")
    # model: GPT = keras._tf_keras.keras.models.load_model("./models/model_v_alpha_1.8.1_pre_train_256emb_6tr_6h1763305151.4034777.keras1763307219.8475053.keras")
    # model: GPT = keras.models.load_model("./models/model_v_alpha_1.8.3_non_pre_train_256emb_4tr_6h1763480600.1235693.keras1763492063.1042597.keras")
    # model.name_of_model = "model_v_alpha_1.8.3_pre_train_fine_tuning_256emd_4tr_6h"
    # model.name_of
    # model: GPT = keras.models.load_model("./models/model_v_alpha_1.8.3_non_pre_train_256emb_2tr_2h1763395283.7493973.keras1763395993.9089189.keras")
    # model = keras._tf_keras.keras.models.load_model("./models/model_v_alpha_1.8.3_non_pre_train_256emb_2tr_2h1763397044.3703222.keras1763398561.8316736.keras")
    # weights = model.get_weights()
    # model = GPT('model_v_alpha_1.8.3_non_pre_train_256emb_4tr_6h', 256, seq_lenght, vocab_size, 0.4)
    # model = GPT('model_v_alpha_1.8.3_non_pre_train_256emb_4tr_6h', 256, seq_lenght, vocab_size, 0.4)
    # model.set_weights(weights)
    
    # model = GPT('model_special_USSR_edition_v1.1_word', 768, seq_lenght, vocab_size, 0.1)
    model = GPT('model_v_alpha_1.8.4_pre_train_fine_tuning_256emd_4tr_6h', 256, seq_lenght, vocab_size, 0.3)
    # model = GPT('model_v_alpha_1.6.1.4_re5_dataset', 512, seq_lenght, vocab_size, 0.2)
    # model.load_weights("./models/model_weights_40_epochs.weights.h5")
    # optimizer = Adam(learning_rate=0.00001, clipnorm=1.0)  # вместо 1e-3 или больше
    # lr_schedule = keras.optimizers.schedules.CosineDecay(
    # 5e-4, 100000  # 100k шагов
    # )
    # optimizer = AdamW(
    #     # learning_rate=1e-4,
    #     lr_schedule,
    #     weight_decay=1e-5,  # 
    #     beta_1=0.9,
    #     # clipnorm=1.0,
    #     beta_2=0.98,
    #     epsilon=1e-8,
    #     global_clipnorm= 1.0 # Градиент
    #     )

    
    optimizer = AdamW(
    learning_rate=1e-4,  
    epsilon=1e-8,
    beta_1=0.9,
    beta_2=0.98,
    weight_decay=0.01,
    global_clipnorm=1.0
    )
    # loss_fn = custom_loss()
    # optimizer = RMSprop(
    #     learning_rate=1e-4,
    #     weight_decay=0.01,  # 
    #     # beta_1=0.9,
    #     clipnorm=1.0,
    #     # beta_2=0.98
    #     )
    # optimizer = Adafactor(
    #     learning_rate=1e-4,
    #     weight_decay=0.01,  # 
    #     # beta_1=0.9,
    #     clipnorm=1.0,
    #     # beta_2=0.98
    #     )

    model.compile(
        optimizer=optimizer,
        # loss=custom_loss_2, 
        loss=weighted_sparse_categorical_crossentropy, 
        # loss=custom_loss, 
        # loss=frequency_penalized_loss, 
        # loss=SparseCategoricalCrossentropy, 
        # loss="sparse_categorical_crossentropy", 
        # loss_fn = FocalLoss(alpha=0.5, gamma=2.0, vocab_size=13669)

        # loss="sparse_categorical_crossentropy", 
        metrics=[ 
            SparseCategoricalAccuracy(name='accuracy'), 
            SparseTopKCategoricalAccuracy(k=5, name='top5_accuracy'), 
            SparseTopKCategoricalAccuracy(k=10, name='top10_accuracy'),
            ],
        jit_compile=True
        )
    #         # 'accuracy', 'perplexity', 
    model.get_config()
    model.summary()
    trainer = GPTTrainer(model, dataset, tokenizer, 0.1)

    print(trainer.fit())
    predict.main(model)


if __name__ == '__main__':
    main()

