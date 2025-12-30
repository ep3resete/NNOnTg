import time
# import os
import json
from keras._tf_keras.keras.optimizers import Adam, AdamW, Nadam
from keras._tf_keras import keras
from keras._tf_keras.keras.losses import SparseCategoricalCrossentropy
from keras._tf_keras.keras.metrics import SparseCategoricalAccuracy, SparseTopKCategoricalAccuracy, Mean
from .callbacks import LoggingModelCheckpoint, PredictionAnalyzer
from Model import GPT
from Data import Dataset
import tensorflow as tf
# from src.Model import GPT
# from src.Data import Dataset
# from Model import GPTWithGenerator

# Установить лимит в 4GB (или больше)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], False)  # отключение growth
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6144)]  # 6 GB
        )
        print("Лимит видеопамяти установлен: 6GB")
    except RuntimeError as e:
        print(f"Ошибка: {e}")


class GPTTrainer:
    # Открыетие файла с конфигов для путей
    with open("./config/PathConfig.json", 'r', encoding='utf-8') as paths_file:
        paths_file = json.load(paths_file) # Все основные пути 
        path_to_data_config = paths_file["DataConfig"] # Путь к конфигу для данных
        path_to_training_config = paths_file["TrainingConfigPath"] # Путь к конфигу для обучения
        path_to_save_callbacks_of_models = paths_file["SaveModelsPath"] # Путь к сохранению моделей
        path_to_json_logs = paths_file["PathToLogs"]
        
        # path_to_input_config = paths_file["InputConfig"] # Путь к конфигу входа
        # path_to_folder_with_datasets = paths_file["SaveDatasetPath"] # Путь к папке сохранения датасета
        del paths_file # Удаление неиспользуемой переменной
    
    # Открытие файла с конфигом для данных
    with open(path_to_data_config, 'r', encoding='utf-8') as config_data_file:
        data_config = json.load(config_data_file) # Конфиг для данных
        path_to_file_with_row_tokenized_data = data_config["path_to_tokenized_file"] # Путь до файла с сырыми токенизированными данными
        dataset_config = data_config["dataset_config"] # Конфиг датасета
        path_to_dataset_json = dataset_config["path_to_dataset_json"] # Путь до файла датасета в json формате
        shuffle_buffer_size = dataset_config["shuffle_buffer_size"]
        batch_size = dataset_config["batch_size"]
        del data_config # Удаление лишней переменной
    
    # Открытие файла с конфигом для обучения
    with open(path_to_training_config, 'r', encoding='utf-8') as config_training_file:
        training_config = json.load(config_training_file) # Конфиг для данных
        epochs_for_fit = training_config["epochs"]
        del training_config # Удаление лишней переменной

    def __init__(self, model: GPT, dataset: Dataset, tokenizer, val_part: float=0.0,):
        """ 
        Класс для тренировки модели GPTWithGenerator
        model - Модель класса GPTWithGenerator
        dataset - Датасет класса Dataset
        val_part - Часть от датасета, которая уйдет в валидационные значения. Указывать от 0.0 до 1.0. Если не нужно валидационных данных - указывать 0
        """
        self.model: GPT = model
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.val_part = val_part
        
        # Оптимизатор и функция потерь
        # self.optimezer = Adam(learning_rate=0.001)
        self.optimizer = AdamW(
            learning_rate=3e-4,
            weight_decay=0.001,  # 
            beta_1=0.9,
            clipnorm=1.0,
            beta_2=0.98
            )
        self.loss_fn = SparseCategoricalCrossentropy(from_logits=False)

        self._init_metrics()

    def _init_metrics(self):
        """Инициализация метрик внутри трейнера"""
        # Основные метрики
        self.train_loss_metric = Mean( name='train_loss')
        self.train_accuracy_metrics = [ 
            SparseCategoricalAccuracy(name='accuracy'), 
            SparseTopKCategoricalAccuracy(k=5, name='top5_accuracy'), 
            SparseTopKCategoricalAccuracy(k=10, name='top10_accuracy'),
            ],
        self.val_loss_metric = Mean(name='val_loss')
        self.val_accuracy_metrics = [ 
            SparseCategoricalAccuracy(name='val_accuracy'), 
            SparseTopKCategoricalAccuracy(k=5, name='val_top5_accuracy'), 
            SparseTopKCategoricalAccuracy(k=10, name='val_top10_accuracy'),
            ]
        
        self.train_perplexity_metric = Mean(name='train_perplexity')
        self.val_perplexity_metric = Mean(name='val_perplexity')

    @tf.function
    def compute_loss(self, x_batch, y_batch, training=True):
        """
        Вычисление потерь для батча
        
        Args:
            x_batch: Входные данные
            y_batch: Целевые значения
            training: Флаг обучения (влияет на dropout/batch norm)
        
        Returns:
            loss: Значение функции потерь (энтропия!)
            predictions: Предсказания модели
        """
        predictions = self.model(x_batch, training=training)
        loss = self.loss_fn(y_batch, predictions)  
        
        # Добавление регуляризационных потерь, если есть
        if self.model.losses:
            loss += tf.add_n(self.model.losses)
            
        return loss, predictions
    
    @tf.function
    def compute_perplexity(self, loss):
        """Вычисление перплексии из потерь"""
        return tf.exp(loss)

    @tf.function
    def train_step(self, x_batch, y_batch):
        """
        Один шаг обучения для одного батча
        
        Args:
            x_batch: Входные данные
            y_batch: Целевые значения
        
        Returns:
            loss: Значение потерь для этого батча
        """
        with tf.GradientTape() as tape:
            loss, predictions = self.compute_loss(x_batch, y_batch, training=True)
        
        # Вычисление градиентов и обновление весов
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Обновление всех метрик трейнера
        self.train_loss_metric.update_state(loss)
        map(lambda x: x.update_state(y_batch, predictions), self.train_accuracy_metrics)
        
        # Обновление дополнительных метрик
        perplexity = self.compute_perplexity(loss)
        self.train_perplexity_metric.update_state(perplexity)
        
        return loss
    
    @tf.function
    def val_step(self, x_batch, y_batch):
        """
        Один шаг валидации для одного батча
        
        Args:
            x_batch: Входные данные
            y_batch: Целевые значения
        
        Returns:
            loss: Значение потерь для этого батча
        """
        loss, predictions = self.compute_loss(x_batch, y_batch, training=False)
        
        # Обновление всех метрик валидации трейнера
        self.val_loss_metric.update_state(loss)
        map(lambda x: x.update_state(y_batch, predictions), self.val_accuracy_metrics)
        
        # Обновление дополнительных метрик
        perplexity = self.compute_perplexity(loss)
        self.val_perplexity_metric.update_state(perplexity)
        
        return loss
    
    @tf.function
    def reset_metrics(self):
        """Сброс ВСЕХ метрик трейнера"""
        self.train_loss_metric.reset_state()
        map(lambda x: x.reset_state(), self.train_accuracy_metrics)
        self.val_loss_metric.reset_state()
        map(lambda x: x.reset_state(), self.val_accuracy_metrics)

    
    @tf.function
    def get_metrics(self):
        """
        Получение текущих значений ВСЕХ метрик из трейнера
        
        Returns:
            dict: Словарь со всеми значениями метрик
        """
        metrics = {
            'train_loss': self.train_loss_metric.result().numpy(),
            'train_accuracy': self.train_accuracy_metric.result().numpy(),
            'val_loss': self.val_loss_metric.result().numpy(),
            'val_accuracy': self.val_accuracy_metric.result().numpy(),
            'train_perplexity': self.train_perplexity_metric.result().numpy(),
            'val_perplexity': self.val_perplexity_metric.result().numpy(),
        }
        
        return metrics
    
    @tf.function
    def custom_fit(self, initial_epoch=0, save_freq=5, log_freq=100):
        """
        Кастомный метод обучения модели
        
        Args:
            initial_epoch: Начальная эпоха (для продолжения обучения)
            save_freq: Частота сохранения модели (в эпохах)
            log_freq: Частота логирования (в батчах)
        
        Returns:
            history: История обучения
        """
        print("Запуск кастомного обучения...")
        
        # Подготовка датасетов
        dataset = self.dataset.prepare_dataset_to_tf(self.shuffle_buffer_size, fix_length=True)
        train_batches = dataset.cardinality().numpy()
        
        # Разделение на train/validation
        if (self.val_part > 0) and (self.val_part < 1):
            train_size = int(train_batches * (1 - self.val_part))
            train_dataset = dataset.take(train_size)
            val_dataset = dataset.skip(train_size)
            val_batches = val_dataset.cardinality().numpy()
            print(f"Данные разделены: {train_size} батчей для обучения, {val_batches} для валидации")
        else:
            train_dataset = dataset
            val_dataset = None
            print(f"Используются все {train_batches} батчей для обучения")
        
        # История обучения - добавляем все метрики
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'train_perplexity': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_perplexity': [],
            'epoch_time': []
        }
        
        # Основной цикл обучения по эпохам
        for epoch in range(initial_epoch, self.epochs_for_fit):
            print(f"\nЭпоха {epoch + 1}/{self.epochs_for_fit}")
            start_time = time.time()
            
            # Сброс метрик в начале каждой эпохи
            self.reset_metrics()
            
            # Обучение
            batch_num = 0
            for batch in train_dataset:
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    x_batch, y_batch = batch
                else:
                    x_batch = batch
                    y_batch = batch
                
                loss = self.train_step(x_batch, y_batch)
                
                batch_num += 1
                if batch_num % log_freq == 0:  # Логирование каждые N батчей
                    current_metrics = self.get_metrics()
                    print(f"  Батч {batch_num}, Потери: {loss:.4f}, "
                          f"Средние потери: {current_metrics['train_loss']:.4f}, "
                          f"Точность: {current_metrics['train_accuracy']:.4f}, "
                          f"Перплексия: {current_metrics['train_perplexity']:.4f}")
            
            # Валидация
            if val_dataset is not None:
                print("Валидация")
                val_batch_num = 0
                for val_batch in val_dataset:
                    if isinstance(val_batch, (list, tuple)) and len(val_batch) == 2:
                        x_val, y_val = val_batch
                    else:
                        x_val = val_batch
                        y_val = val_batch
                    
                    self.val_step(x_val, y_val)
                    val_batch_num += 1
            
            # Расчет времени эпохи
            epoch_time = time.time() - start_time
            
            # Получение финальных метрик эпохи
            final_metrics = self.get_metrics()
            
            # Сохранение в историю всех метрик
            history['train_loss'].append(final_metrics['train_loss'])
            history['train_accuracy'].append(final_metrics['train_accuracy'])
            history['train_perplexity'].append(final_metrics['train_perplexity'])
            history['epoch_time'].append(epoch_time)
            
            if val_dataset is not None:
                history['val_loss'].append(final_metrics['val_loss'])
                history['val_accuracy'].append(final_metrics['val_accuracy'])
                history['val_perplexity'].append(final_metrics['val_perplexity'])
            
            # Вывод прогресса
            print(f"  Время эпохи: {epoch_time:.2f}с")
            print(f"  Потери обучения: {final_metrics['train_loss']:.4f}")
            print(f"  Точность обучения: {final_metrics['train_accuracy']:.4f}")
            print(f"  Перплексия обучения: {final_metrics['train_perplexity']:.4f}")
            if val_dataset is not None:
                print(f"  Потери валидации: {final_metrics['val_loss']:.4f}")
                print(f"  Точность валидации: {final_metrics['val_accuracy']:.4f}")
                print(f"  Перплексия валидации: {final_metrics['val_perplexity']:.4f}")
            
            # Сохранение модели
            if (epoch + 1) % save_freq == 0 or (epoch + 1) == self.epochs_for_fit:
                model_path = f"{self.path_to_save_callbacks_of_models}{self.model.name_of_model}/epoch_{epoch + 1}"
                self.model.save(model_path)
                print(f"  Модель сохранена: {model_path}")
        
        print("Обучение завершено!")
        return history

    def fit(self):
        """ Метод для обучения модели """
        dataset = self.dataset.prepare_dataset_to_tf(False, fix_length=True) # Подготовка данных для обучения
        train_batches = dataset.cardinality().numpy()
        val_dataset = None # Данные для валидации. Изначально специально None, так как не факт, что он изменится
        # Проверка на то, нужно ли переместить часть данных в датасет для валидации
        if (self.val_part > 0) and (self.val_part < 1): 
            train_dataset = dataset.take(int(train_batches * (1 - self.val_part))) # Данные для обучения
            train_dataset = train_dataset.shuffle(self.shuffle_buffer_size)
            val_dataset = dataset.skip(int(train_batches * self.val_part)) # Данные для валидации
            val_dataset = val_dataset.shuffle(self.shuffle_buffer_size)
        else:
            train_dataset = dataset # Данные для обучения (весь датасет)
        
        callbacks = [] # Список с коллбэками
        lr_scheduler = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6
        )
        callbacks.append(lr_scheduler)

        
        
        analyzer = PredictionAnalyzer(
            val_dataset, self.tokenizer, num_samples=20
        )
        callbacks.append(analyzer)

        
        callbacks.append(
            LoggingModelCheckpoint( # Коллбэк 
                self.tokenizer,
                self.path_to_save_callbacks_of_models + self.model.name_of_model, # Путь для сохранения модели
                model_name=self.model.name_of_model, # Имя модели
                path_to_log_json=self.path_to_json_logs, # Путь к сохранению логов
                save_weights_only=False, # Сохраняются только веса. Так меньше места займёт
            )
        )
        
        # Само обучение
        history = self.model.fit(
            train_dataset, # Датасет для обучения
            epochs=self.epochs_for_fit, # Количество эпох
            validation_data=val_dataset, # Датасет для валидации
            callbacks=callbacks, # Коллбэки
            verbose=1,
            initial_epoch=4,

        )
        
        return history
