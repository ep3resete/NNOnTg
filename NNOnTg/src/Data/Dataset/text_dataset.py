import tensorflow as tf
from .base_dataset import BaseDataset
from Data import GPTTokenizer
from typing import Any, List, Dict, Tuple, Optional, Union, Callable

class TextDataset(BaseDataset):
    def __init__(self, data_path: str, tokenizer: GPTTokenizer, max_length: Optional[int] = None):
        self.max_length: Optional[int] = max_length or self.seq_length
        self.texts = self.from_json_file(self.from_json_file(self.path_to_file_with_raw_tokenized_texts))
        super().__init__(data_path, tokenizer)

    # @classmethod
    # def from_data(cls, data: List[Union[Dict[str, str], str]]) -> 'TextDataset':
    #     """
    #     Создает датасет из данных текстов
        
    #     Args:
    #         data: Список текстов или словарей с текстами
            
    #     Returns:
    #         Экземпляр TextDataset
            
    #     Raises:
    #         ValueError: Если не найдено валидных текстовых данных
    #     """
    #     texts: List[str] = cls._extract_texts(data)
    #     variant_tensor: tf.Tensor = tf.constant(texts)
    #     return cls(variant_tensor)
    
    # def from_data(self, data):
    #     for text in self.texts:
    #     pass

    def _create_variant_tensor(self) -> tf.Tensor:
        """Создание variant_tensor для простых текстов"""
        # Извлечение текстов
        texts = self._extract_texts()
        # Создение tf.data.Dataset
        dataset = tf.data.Dataset.from_tensor_slices(texts)
        # Применение пайплайна обработки
        self._apply_processing_pipeline(dataset)
        return 
        
        # return processed_dataset._variant_tensor


    def _extract_texts(self) -> List[str]:
        """Извлечение текстов из данных"""
        texts: List[str] = []
        for item in self.texts:
            if isinstance(item, dict) and 'text' in item:
                texts.append(item['text'])
            elif isinstance(item, str):
                texts.append(item)
        
        if not texts:
            raise ValueError("No valid text data found")
        
        return texts
    
    def _apply_processing_pipeline(self, dataset):
        """Пайплайн обработки для простых текстов"""
        def _tokenize_and_convert(text: tf.Tensor) -> tf.Tensor:
            """Обертка для твоего токенайзера"""
            # 1. Конвертируем tf.Tensor в Python строку
            text_str = text.numpy().decode('utf-8')
            
            # 2. Применяем твой токенайзер (возвращает list)
            tokens_list = self.tokenizer.tokenize_text(text_str)
            
            # 3. Конвертируем обратно в tf.Tensor
            return tf.constant(tokens_list, dtype=tf.int32)
        
        def _pad_sequences(tokens: tf.Tensor) -> tf.Tensor:
            """Обрезка до максимальной длины"""
            return tokens[:self.max_length]
        
        # Для языкового моделирования создаем пары (input, target)
        def _create_lm_pairs(tokens: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
            """Создание пар для языкового моделирования"""
            inputs = tokens[:-1]
            targets = tokens[1:]
            return inputs, targets
        
        return (dataset
                .map(_tokenize_text)
                .map(_pad_sequences)
                .map(_create_lm_pairs)
                .shuffle(self.shuffle_buffer_size)
                .padded_batch(
                    self.batch_size,
                    padded_shapes=([None], [None]),
                    padding_values=(0, 0),
                    drop_remainder=True
                )
                .prefetch(tf.data.AUTOTUNE))
