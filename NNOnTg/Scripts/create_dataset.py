"""
При запуске скрипта создастся и автоматически сохранится датасет.
Обязательно нужно, чтобы перед ним был запущен скрипт tokenize_dataset
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, 'src')

# Удаление всех существующих путей и добавление пути с src
sys.path = [src_path] + [p for p in sys.path if p != src_path]


import json
import time
import pickle
# from Data import Dataset
from Data import Dataset

def create_and_save_dataset(name_of_dataset, dataset_save_path, save_to_json, shuffle_buffer_size):
    """ Функция для создания и сохранения датасета """
    dataset = Dataset() # Создание объекта датасета (Создание датасета из файла с сырыми токенизированными данными)
    dataset.learneble_data_tf = dataset.prepare_dataset_to_tf(shuffle_buffer_size)
    name_of_file_dataset = name_of_dataset + "-" +time.strftime("%Y-%m-%d_%H-%M-%S") + ".pkl" # Имя для файла датасета
    path_to_save_dataset = dataset_save_path + name_of_file_dataset 
    # Открытие файла для сохранения датасета
    # dataset.save(path_to_save_dataset)

    # with open(path_to_save_dataset, 'wb') as tokenizer_file:
    #     pickle.dump(dataset, tokenizer_file)
    if save_to_json:
        dataset.save_dataset_to_json()

def main():
    # Открытие конфига с путями 
    with open("./config/PathConfig.json") as path_config_file:
        path_config = json.load(path_config_file) # Конфиг с путями
        dataset_save_path = path_config["SaveDatasetPath"] # Путь к папке с датасетами
        path_of_data_config_file = path_config["DataConfig"] # Конфиг для данных
        del path_config # Удаление переменной с путями
    
    # Открытие файла с конфигом для данных
    with open(path_of_data_config_file) as data_config_file:
        config_data = json.load(data_config_file) # Конфиг для данных
        dataset_config = config_data["dataset_config"]
        name_of_dataset_file = dataset_config["name_of_dataset"] # Имя для датасета
        save_dataset_to_json = dataset_config["save_to_json"]
        shuffle_buffer_size = dataset_config["shuffle_buffer_size"]
    create_and_save_dataset(name_of_dataset_file, dataset_save_path, save_dataset_to_json, shuffle_buffer_size) # Создание и сохранение датасета

if __name__ == '__main__':
    main()
    # print(1)