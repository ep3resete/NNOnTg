import tensorflow as tf
import json
import numpy as np
from Data import BPETokenizer
from Model import GPT
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from .strategies import STRATEGIES



class GPTPredictor:
    """ Предиктор для GPT с разными стратегиями 
    """
    
    with open("config/PathConfig.json", 'r', encoding='utf-8') as path_config_file:
        path_to_input_file_config = json.load(path_config_file)["InputConfig"]

    with open(path_to_input_file_config, 'r', encoding='utf-8') as input_settings_file:
        input_settings = json.load(input_settings_file)
        seq_length = input_settings["seq_length"]

    def __init__(self, model: GPT, tokenizer: BPETokenizer, return_with_promt: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.return_with_promt = return_with_promt
        self._strategy_instances = {} # Словарь [<название стратегии>: <объект стратегии>] с различными стратегиями генерации. (Сейчас не очень нужно, просто заготовка на будущее)
        self.prompt = ''


    def generate(self, prompt, max_tokens=20, strategy='top_p', **kwargs):
        """ 
        Генерация с выбором стратегии 
        
        Аргументы:
            strategy:
                - "greedy": всегда самый вероятный токен
                - "sample": сэмплирование с температурой
                - "top_k": топ-K сэмплирование  
                - "top_p": топ-P (nucleus) сэмплирование
            **kwargs:
                - temperature: для sample/top_k/top_p (default: 1.0)
                - top_k: для top_k стратегии (default: 50)
                - top_p: для top_p стратегии (default: 0.9)
        """
        if self.prompt == '':
            prompt = "<user>" + prompt + "<eos><bot>"
        else:
            prompt = self.prompt + "<user>" + prompt + "<eos><bot>"
        self.prompt = prompt
        tokens = self.tokenizer.tokenize_text(prompt) # Токенизация промпта
        tokens = [x for x in tokens if x != 9936]
        # tokens = [0] * (self.seq_length - len(tokens)) + tokens
        padded_tokens = pad_sequences([tokens], maxlen=self.seq_length, padding='pre', truncating='pre', value=0)[0]
        strategy_obj = self._get_strategy(strategy, **kwargs) # Создание объекта стратегии

        generated_tokens = strategy_obj.generate(padded_tokens, max_tokens, self.return_with_promt, **kwargs) # Вся сгенерированная последовательность 
        if generated_tokens[-1] != self.tokenizer.special_tokens['<eos>']:
            np.append(generated_tokens, self.tokenizer.special_tokens['<eos>'])
        generated_text = self.tokenizer.detokenize_text(generated_tokens).replace('</w>', ' ')
            # generated_text[-1] += ("<eos>")
        print(
            f"Сообщение пользователя: {prompt}\n",
            f"Сообщение пользователя в токенах: {tokens}\n",
            f"Ответ сети: {generated_text}\n",
            f"Ответ сети в ткоенах: {generated_tokens}\n" + '-' * 50 + '\n'
              )
        self.prompt += generated_text
        return generated_text
    

    def _get_strategy(self, strategy_name: str, **kwargs):
        """ 
        Вспомогательный метод для получения объекта стратегии. Так же она добавляется в список self._strategy_instances 
        Аргументы:
            strategy_name - имя стратегиии
        **kwargs:
                - temperature: для sample/top_k/top_p (default: 1.0)
                - top_k: для top_k стратегии (default: 50)
                - top_p: для top_p стратегии (default: 0.9)
        """

        if strategy_name not in self._strategy_instances:
            if strategy_name not in STRATEGIES:
                raise ValueError(f"Несуществующая стратегия: {strategy_name}")
            
            strategy_class = STRATEGIES[strategy_name]
            self._strategy_instances[strategy_name] = strategy_class(self.model, self.tokenizer, **kwargs)

            return self._strategy_instances[strategy_name]
        return self._strategy_instances[strategy_name]
