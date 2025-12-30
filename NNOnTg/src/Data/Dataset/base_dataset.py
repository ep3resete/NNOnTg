# from keras._tf_keras.keras.datasets import  
import tensorflow as tf
import json
from collections import OrderedDict, Counter
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Tuple, Optional, Union, Callable
from Data import GPTTokenizer

class BaseDataset(tf.data.Dataset, ABC):
     # Открыетие файла с конфигов для путей
    with open("./config/PathConfig.json", 'r', encoding='utf-8') as paths_file:
        paths_file = json.load(paths_file) # Все основные пути 
        path_to_data_config = paths_file["DataConfig"] # Путь к конфигу для данных
        path_to_input_config = paths_file["InputConfig"] # Путь к конфигу входа
        path_to_folder_with_datasets = paths_file["SaveDatasetPath"] # Путь к папке сохранения датасета
        del paths_file # Удаление неиспользуемой переменной
    
    # Открыетие файла с конфигов для входа
    with open(path_to_input_config, 'r', encoding='utf-8') as config_input_file: 
        input_config = json.load(config_input_file) # Конфиг для входа
        seq_length = input_config["seq_length"] # Максимальная входная последовательность
        del input_config # Удаление лишней переменной

    # Открытие файла с конфигом для данных
    with open(path_to_data_config, 'r', encoding='utf-8') as config_data_file:
        data_config = json.load(config_data_file) # Конфиг для данных
        path_to_file_with_row_tokenized_dialogs = data_config["path_to_tokenized_file"] # Путь до файла с сырыми токенизированными данными
        path_to_file_with_row_tokenized_texts = data_config["path_to_tokenized_file"] # Путь до файла с сырыми токенизированными данными
        vocab_size = data_config["vocab_size"]
        dataset_config = data_config["dataset_config"] # Конфиг датасета
        path_to_dataset_json = dataset_config["path_to_dataset_json"] # Путь до файла датасета в json формате
        shuffle_buffer_size = dataset_config["shuffle_buffer_size"]
        batch_size = dataset_config["batch_size"]
        # path_to_simple_texts = data_config.get("path_to_simple_texts", "")
        # path_to_dialogues = data_config.get("path_to_dialogues", "")
        del data_config # Удаление лишней переменной

    def __init__(self, variant_tensor: tf.Tensor, tokenizer: GPTTokenizer) -> None:
        """
        Инициализация базового датасета
        
        Args:
            variant_tensor: Тензор, содержащий данные датасета
            tokenizer: Токенайзер класса GPTTokenizer
        """
        self.tokenizer = tokenizer
        super().__init__(variant_tensor)
        self._setup_automatic_processing()

    @abstractmethod
    def _setup_automatic_processing(self) -> None:
        """Настройка автоматической обработки (должен быть реализован в подклассах)"""
        pass
    
    @abstractmethod
    def _preprocess_data(self, data: tf.Tensor) -> tf.Tensor:
        """Предобработка данных (должен быть реализован в подклассах)"""
        pass
    
    def _pad_or_truncate(self, tokens_tensor: tf.Tensor) -> tf.Tensor:
        """
        Обрезка или паддинг до seq_length
        
        Args:
            tokens_tensor: Тензор с токенами
            
        Returns:
            Тензор обрезанный/дополненный до seq_length
        """
        return tokens_tensor[:self.seq_length]
    

    def _create_basic_pipeline(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """
        Базовый пайплайн обработки
        
        Args:
            dataset: Исходный датасет
            
        Returns:
            Обработанный датасет с примененными преобразованиями
        """
        return (dataset
                .shuffle(self.shuffle_buffer_size)
                .batch(self.batch_size)
                .prefetch(tf.data.AUTOTUNE))
    

    @classmethod
    def from_json_file(cls, file_path: Optional[str] = None) -> 'BaseDataset':
        """
        Создает датасет из JSON файла
        
        Args:
            file_path: Путь к JSON файлу. Если None, используется путь из конфига
            
        Returns:
            Экземпляр датасета
            
        Raises:
            FileNotFoundError: Если файл не найден
            JSONDecodeError: Если файл содержит некорректный JSON
        """
        if file_path is None:
            file_path = cls.path_to_dataset_json
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data: List[Any] = json.load(f)
        
        return cls.from_data(data)
    

    @classmethod
    @abstractmethod
    def from_data(cls, data: List[Any]) -> 'BaseDataset':
        """
        Создает датасет из данных (должен быть реализован в подклассах)
        
        Args:
            data: Список данных для создания датасета
            
        Returns:
            Экземпляр датасета
        """
        pass