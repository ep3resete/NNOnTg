import numpy as np
from Model import GPT
from Data import BPETokenizer
import tensorflow as tf
from .base import BaseStrategy


class TopKStrategy(BaseStrategy):
    def __init__(self, model: GPT, tokenizer: BPETokenizer):
        """ 
        Аргументы:
            model: GPT - обученная GPT-модель
            tokenizer: BPETokenizer - обученный токенизатор
        """
        super().__init__(model, tokenizer)
    
    def generate_next_token(self, tokens: np.ndarray, top_k: int = 10, temperature: float = 1.0, **kwargs) -> int:
        """
        Метод для определения следующего токена с использованием top-k фильтрации.

        Аргументы:
            tokens: np.ndarray - Токены промпта, который подаётся на вход модели
            top_k: int - Количество токенов с наибольшей вероятностью для выборки
            temperature: float - Коэффициент температуры для управления случайностью
            
        Возвращает:
            int - Следующий сгенерированный токен
        """
        # Проверка корректности входных данных
        if not isinstance(top_k, int) or top_k < 0:
            raise ValueError("top_k должен быть неотрицательным целым числом")
        
        if temperature <= 0:
            raise ValueError("temperature должна быть положительным числом")
        
        if len(tokens) == 0:
            raise ValueError("Массив tokens не может быть пустым")

        # Логиты от модели
        input_tensor = tf.constant([tokens], dtype=tf.int32)
        logits = self.model.call(input_tensor, training=False)
        
        # Логиты для последнего токена в последовательности
        next_token_logits = logits[-1, :]
        
        # top-k фильтрация
        filtered_logits = self._top_k_filter(next_token_logits, top_k)

        # Температура
        if temperature != 1.0:
            filtered_logits = filtered_logits / temperature
        
        # Преобразование в вероятности с помощью softmax
        probabilities = self._softmax(filtered_logits)
        
        # Выбор следующего токена на основе вероятностей
        next_token = self._sample_from_probabilities(probabilities)
        
        return next_token
    
    def _top_k_filter(self, logits: tf.Tensor, top_k: int) -> tf.Tensor:
        """
        Фильтрует логиты, оставляя только top-k токенов с наибольшими значениями.
        
        Аргументы:
            logits: tf.Tensor - Логиты модели
            top_k: int - Количество токенов для сохранения
            
        Возвращает:
            tf.Tensor - Отфильтрованные логиты
        """
        if top_k <= 0 or top_k >= logits.shape[0]:
            return logits
        
        values, _ = tf.math.top_k(logits, k=top_k)
        min_value = tf.reduce_min(values)
        
        # Создание маски для логитов, которые меньше k-го наибольшего
        mask = logits >= min_value
        
        # Применение маски: остабтся только top-k логитов, остальные -> -inf
        filtered_logits = tf.where(
            mask,
            logits,
            tf.fill(logits.shape, -float('inf'))
        )
        
        return filtered_logits
    
    def _softmax(self, x: tf.Tensor) -> np.ndarray:
        """
        Вычисляет softmax для тензора TensorFlow.
        
        Аргументы:
            x: tf.Tensor - Входной тензор
            
        Возвращает:
            np.ndarray - Вероятности после softmax
        """
        if x.shape.num_elements() == 0:
            raise ValueError("Входной тензор для softmax не может быть пустым")
            
        probabilities = tf.nn.softmax(x).numpy()
        
        return probabilities
    
    def _sample_from_probabilities(self, probabilities: np.ndarray) -> int:
        """
        Выбирает токен на основе распределения вероятностей.
        
        Аргументы:
            probabilities: np.ndarray - Распределение вероятностей
            
        Возвращает:
            int - Выбранный токен
        """
        if probabilities.size == 0:
            raise ValueError("Массив вероятностей не может быть пустым")
        
        # Проверка, что вероятности не все нулевые (после top-k фильтрации)
        if np.all(probabilities == 0):
            raise ValueError("Все вероятности равны нулю после фильтрации")
        
        # Проверка, что вероятности валидны (не содержат NaN или inf)
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("Вероятности содержат нечисловые значения")
        
        # Нормализация не требуется, так как softmax уже возвращает нормализованные вероятности
        # Выбор токена на основе распределения вероятностей
        return np.random.choice(len(probabilities), p=probabilities)