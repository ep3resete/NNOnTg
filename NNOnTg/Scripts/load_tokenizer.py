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

# from Data import GPTTokenizer
import json
import pickle

def load_tokenizer():
    """ Функция для создания и сохранения токенайзера """
    with open("./config/PathConfig.json") as path_config_file:
        path_config = json.load(path_config_file) # Конфиг с путями
        current_tokenizer_path = path_config["SubwordTokenizerPath"]
        del path_config # Удаление переменной с путями
    
    # Открытие файла для загрузки токенайзера
    with open(current_tokenizer_path, 'rb') as tokenizer_file:
        tokenizer = pickle.load(tokenizer_file)
    return tokenizer


def main():
    # Открытие конфига с путями 
    # Открытие файла с конфигом для данных
    # with open(path_of_data_config_file) as data_config_file:
    #     config_data = json.load(data_config_file) # Конфиг для данных
    #     name_of_tokenizer_file = config_data["tokenizer_config"]["name_of_tokenizer"] # Имя для токенайзера
    
    # create_and_save_tokenizer(name_of_tokenizer_file, tokenizer_save_path) # Создание и сохранение токенайзера
    load_tokenizer()

if __name__ == "__main__":
    main()