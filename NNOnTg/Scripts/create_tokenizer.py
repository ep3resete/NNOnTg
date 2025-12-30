import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, 'src')

# Удаление всех существующих путей и добавление пути с src
sys.path = [src_path] + [p for p in sys.path if p != src_path]


from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.metrics import SparseCategoricalAccuracy, SparseTopKCategoricalAccuracy
import tensorflow as tf

from Data import BPETokenizer
import json
import time
import pickle

def create_and_save_tokenizer(name_of_tokenizer_file, tokenizer_save_path):
    """ Функция для создания и сохранения токенайзера """
    gpt_tokenizer = BPETokenizer() # Создание объекта токенайзера (Обучение происходит при инициализации)
    name_of_tokenizer = name_of_tokenizer_file + "-" +time.strftime("%Y-%m-%d_%H-%M-%S") + ".pkl" # Имя для токенайзера
    path_to_save_tokenizer = tokenizer_save_path + name_of_tokenizer
    # Открытие файла для сохранения токенайзера
    with open(path_to_save_tokenizer, 'wb') as tokenizer_file:
        pickle.dump(gpt_tokenizer, tokenizer_file)


def main():
    # Открытие конфига с путями 
    with open("./config/PathConfig.json") as path_config_file:
        path_config = json.load(path_config_file) # Конфиг с путями
        tokenizer_save_path = path_config["SaveTokenizerPath"] # Путь к папке с токенайзерами
        path_of_data_config_file = path_config["DataConfig"] # Конфиг для данных
        del path_config # Удаление переменной с путями
    
    # Открытие файла с конфигом для данных
    with open(path_of_data_config_file) as data_config_file:
        config_data = json.load(data_config_file) # Конфиг для данных
        name_of_tokenizer_file = config_data["tokenizer_config"]["name_of_tokenizer"] # Имя для токенайзера
    
    create_and_save_tokenizer(name_of_tokenizer_file, tokenizer_save_path) # Создание и сохранение токенайзера


if __name__ == "__main__":
    main()