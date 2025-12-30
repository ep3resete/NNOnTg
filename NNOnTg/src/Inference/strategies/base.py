import numpy as np
import json
from Model import GPT
from Data import BPETokenizer
from abc import ABC, abstractmethod # Нужны для того, чтобы создать класс и методы, которые нельзя будет использовать напрямую. Только через наследование


class BaseStrategy(ABC):
    with open("config/PathConfig.json", 'r', encoding='utf-8') as path_config_file:
        path_to_input_file_config = json.load(path_config_file)["InputConfig"]

    with open(path_to_input_file_config, 'r', encoding='utf-8') as input_settings_file:
        input_settings = json.load(input_settings_file)
        seq_length = input_settings["seq_length"]

    def __init__(self, model: GPT, tokenizer: BPETokenizer):
        """ 
        Аргументы:
            model: GPT - обученная GPT-модель
            tokenizer: BPETokenizer - обученный токенизатор
        """
        self.model = model
        self.tokenizer = tokenizer
        # super().__init__()
    
    @abstractmethod
    def generate_next_token(self, tokens, **kwargs):
        """
        Метод для определения следующего токена.

        Аргументы:
            tokens - Токены промпта, который подаётся на вход модели.
        """
        pass

    def is_last_token(self, token):
        """ Метод для проверки того, являетяся ли токен токеном конца в соответствии с токенайзером"""
        eos_token = self.tokenizer.special_tokens["<eos>"]
        return eos_token + 4 == token
    
    def generate(self, input_tokens: np.array, max_length: int=100, return_with_prompt=False, **kwargs):
        """ 
        Метод позволяет генерировать GPT не один токен, а полную последовательность. 
        Генерация останавливается когда нейросеть решает сгенерировать <EOS> (Токен конца генерации), либо когда длина сообщения становистя больше
        
        Аргументы:
            max_length: int - Максимальная длина генерируемого сообщения
            return_with_prompt: bool - Флаг, указывающий на ту, вернуть сгенерированный ответ вместе с промптом или без
            temperature: float - Коэффицент температуры
            top_p: float - коэффицент для top_p метода
        """  

        all_tokens: np.array = input_tokens.copy() # Все токены (Нужно для генерации)
        if return_with_prompt: 
            generated_tokens: np.array = input_tokens.copy() # Когда нужно вернуть ответ вместе с промптом сгенерированными токенами считаются и те токены, что были на входе
        else:
            generated_tokens: np.array = np.array([], dtype=np.int32) # Когда нужно вернуть только ответ сети

        for _ in range(max_length):
            # Поиск следующего токена
            next_token: int = self.generate_next_token(all_tokens, **kwargs)
            # if next_token
            # Добавление его в переменные
            all_tokens = np.append(all_tokens, next_token)
            if len(all_tokens) > self.seq_length:
                all_tokens = np.delete(all_tokens, 0)
            generated_tokens = np.append(generated_tokens, next_token)
            # Проверка на то, является ли сгенерированный токен токеном конца. Если да, то генерация прекращается
            if self.is_last_token(next_token):
                break
        
        return generated_tokens
    
