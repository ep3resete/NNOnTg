import tensorflow as tf
import json
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from collections import OrderedDict, Counter
import numpy as np

class Dataset:
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
        path_to_file_with_row_tokenized_data = data_config["path_to_tokenized_file"] # Путь до файла с сырыми токенизированными данными
        vocab_size = data_config["vocab_size"]
        dataset_config = data_config["dataset_config"] # Конфиг датасета
        path_to_dataset_json = dataset_config["path_to_dataset_json"] # Путь до файла датасета в json формате
        shuffle_buffer_size = dataset_config["shuffle_buffer_size"]
        batch_size = dataset_config["batch_size"]
        del data_config # Удаление лишней переменной

    def __init__(self, skip_frequent_tokens=False, word_counts=None, 
                 filter_min_freq=3, filter_max_freq=1500, 
                 filter_rare_prob=0.3, filter_freq_prob=0.4):
        self.dialogs = self.open_file_with_row_data()
        self.learneble_data = self.get_learnable_dataset(skip_frequent_tokens, word_counts)
        
        # Применяем фильтрацию если указаны параметры
        if (filter_min_freq is not None and filter_max_freq is not None) and skip_frequent_tokens:
            self.filter_dataset_by_frequency(
                min_freq=filter_min_freq,
                max_freq=filter_max_freq,
                rare_token_prob=filter_rare_prob,
                frequent_token_prob=filter_freq_prob
            )

    def filter_dataset_by_frequency(self, min_freq=2, max_freq=1000, 
                                  rare_token_prob=0.2, frequent_token_prob=0.4,
                                  keep_special_tokens=True):
        """
        КОРРЕКТНАЯ фильтрация датасета по частоте токенов
        """
        X, Y = self.learneble_data
        
        # Собираем статистику по ВСЕМ токенам в целях (Y)
        token_counts = Counter(Y)
        print(f"Всего уникальных токенов в целях до фильтрации: {len(token_counts)}")
        
        # Определяем специальные токены
        special_tokens = {'<unk>', '<pad>', '<eos>', '<sos>', '<mask>', 'привет', ',', '.', '!', '?'}
        
        # Создаем правила фильтрации для КАЖДОГО целевого токена
        filtering_rules = {}
        for token, count in token_counts.items():
            if token in special_tokens and keep_special_tokens:
                filtering_rules[token] = {'action': 'keep', 'prob': 1.0}
            elif count < min_freq:
                filtering_rules[token] = {'action': 'remove', 'prob': 0.0}
            elif count < min_freq * 3:  # Увеличил диапазон редких токенов
                filtering_rules[token] = {'action': 'keep_prob', 'prob': rare_token_prob}
            elif count > max_freq:
                filtering_rules[token] = {'action': 'remove_prob', 'prob': frequent_token_prob}
            else:
                filtering_rules[token] = {'action': 'keep', 'prob': 1.0}
        
        # Применение фильтрации
        filtered_X = []
        filtered_Y = []
        removed_count = 0
        
        for i, (input_seq, target) in enumerate(zip(X, Y)):
            rule = filtering_rules.get(target, {'action': 'keep', 'prob': 1.0})
            
            should_keep = False
            if rule['action'] == 'keep':
                should_keep = True
            elif rule['action'] == 'keep_prob':
                should_keep = np.random.random() < rule['prob']
            elif rule['action'] == 'remove_prob':
                should_keep = np.random.random() > rule['prob']
            
            if should_keep:
                filtered_X.append(input_seq)
                filtered_Y.append(target)
            else:
                removed_count += 1
        
        # Обновляем обучаемые данные
        self.learneble_data = (filtered_X, filtered_Y)
        
        # Статистика
        print(f"Удалено примеров: {removed_count}")
        print(f"Осталось примеров: {len(filtered_X)}")
        print(f"Уникальных токенов в целях после фильтрации: {len(set(filtered_Y))}")
        
        return self.learneble_data


    def open_file_with_row_data(self):
        """ Метод для открытия файла с сырми токенизированными данными """
        with open(self.path_to_file_with_row_tokenized_data, 'r', encoding='utf-8') as file_with_row_data:
            row_data = json.load(file_with_row_data) # Сырые данные
            return row_data
    
    def prepare_dataset_with_padding(self, data, max_length=None, pad_token_id=0):
        """ Метод для форматирования данных из датасета """
        inputs, targets = data # Разложение всей переменной с данными на вход и таргет
        # Проверка на то, есть ли ограничения на вход сети. По сути, размер вектора контекста. Если нет, то ограничением будет максимальная длина из датасета
        if max_length is None: 
            max_length = len(max(inputs, key=len))
        # Паддинг входных данных
        inputs_padded = pad_sequences(
            inputs, # Изначальные входы 
            maxlen=max_length, # Максимальная длина
            dtype='int32', # Тип данных (инт на 4 байта)
            padding='pre', # Добавлять <PAD> в конец
            truncating='pre', #  Обрезать длину c начала
            value=pad_token_id # Айди пад-токена
        )
        return inputs_padded, targets
    
    def create_tf_dataset_fix_length(self, learneble_padding_data, shuffle=True):
        """ Метод для создания датасета на основе класса tf.data.Dataset
        learneble_padding_data - паддированный токенизированный датасет
        shuffle - (True/False) если да, то перемешивать датасет. Значение для буффера перемешивания берется из файла с конфигом
        """
        inputs_padded, targets = learneble_padding_data # Разбор на входы и таргеты 
        inputs_tensor = tf.constant(inputs_padded, dtype=tf.int32) # Преобразование входов в тензоры

        targets_tensor = tf.constant(targets, dtype=tf.int32) # Преобразование таргетов в тензоры
        
        self.learneble_data_tf = tf.data.Dataset.from_tensor_slices((inputs_tensor, targets_tensor)) # Обучаемые данные в виде датасета типа из TensorFlow
        # Перемешивание, если shuffle = True
        if shuffle: 
            self.learneble_data_tf = self.learneble_data_tf.shuffle(
                buffer_size=self.shuffle_buffer_size, # Размер буфера перемешивания
                reshuffle_each_iteration=True # Перемешивать каждую эпоху
            )
        self.learneble_data_tf = self.learneble_data_tf.batch(batch_size=self.batch_size)
        

    def create_tf_dataset(self, learneble_data, shuffle=True):
        inputs, targets = learneble_data # Разбор на входы и таргеты 
        inputs_tensor = tf.ragged.constant(inputs, dtype=tf.int32) # Преобразование входов в тензоры
        targets_tensor = tf.ragged.constant(targets, dtype=tf.int32) # Преобразование таргетов в тензоры
        
        self.learneble_data_tf = tf.data.Dataset.from_tensor_slices((inputs_tensor, targets_tensor)) # Обучаемые данные в виде датасета типа из TensorFlow
        # Перемешивание, если shuffle = True
        if shuffle: 
            self.learneble_data_tf = self.learneble_data_tf.shuffle(
                buffer_size=self.shuffle_buffer_size, # Размер буфера перемешивания
                reshuffle_each_iteration=True # Перемешивать каждую эпоху
            )
        self.learneble_data_tf = self.learneble_data_tf.map(
            lambda x, y: (
                tf.cast(x, dtype=tf.int32),
                tf.cast(y, dtype=tf.int32)
            ))
        
        def reverse_sequence(x, y):
            return tf.reverse(x, axis=[0]), y
            
        # Разворачиваем последовательности
        self.learneble_data_tf = self.learneble_data_tf.map(reverse_sequence)
        self.learneble_data_tf = self.learneble_data_tf.padded_batch(
            self.batch_size,
            padded_shapes=(
                tf.TensorShape([50]), []),
            padding_values=(0, 0),
            drop_remainder=True
        ).prefetch(tf.data.AUTOTUNE)

        def reverse_batch(x_batch, y_batch):
            return tf.reverse(x_batch, axis=[1]), y_batch
    
        self.learneble_data_tf = self.learneble_data_tf.map(reverse_batch)


    def create_batches(self, batch_size=32):
        """ Метод для сборки датасета в бэтчи. Для этого нужно чтобы сначала был использован метод create_tf_dataset"""
        self.learneble_data_tf = self.learneble_data_tf.batch(batch_size=batch_size)

    def get_learnable_dataset_dialogs(self, ignore_frequent_tokens=False, word_counts: OrderedDict=None):
        """ Метод для создания датасет из сырых токеннизированных данных"""
        X = []  # Входы
        Y = []  # Выходы (цели)

        for dialog in self.dialogs:
            dialog = dialog['dialog'] # Текущий диалог
            current_sequence = [] # Текущая длина. Нужно чтобы добавлять данные "лесенкой"
            
            for j, message in enumerate(dialog):
                if j % 2 == 0:  
                    # Пользователь - добавление всей фразы
                    current_sequence.extend(message)
                else:  
                    # Бот - добавление по токенам
                    for token in message:
                        if not (ignore_frequent_tokens and (word_counts[token] > 1000) and token != 1):
                            # Вход: текущая последовательность
                            X.append(current_sequence.copy())
                            # Выход: следующий токен
                            Y.append(token)
                            # Добавление токена к последовательности
                            current_sequence.append(token)
        
        return X, Y
    
    def prepare_dataset_to_tf(self, shuffle, fix_length=False):
        """ Подготовка датасета к обучению сети (json стиановится тензорами tensorflow). 
        shuffle - размер для буфера перемешивания """
        if fix_length:
            self.learnable_padding_data = self.prepare_dataset_with_padding(self.learneble_data, self.seq_length)
            self.create_tf_dataset_fix_length(self.learnable_padding_data)
        else:
            self.create_tf_dataset(self.learneble_data, shuffle) # Данные для обучениия становятся в виде tf
        return self.learneble_data_tf

    def save_dataset_to_json(self):
        """ Метод сохраняет значения из self.learneble_data в файл dataset.json """
        with open(self.path_to_dataset_json, 'w', encoding='utf-8') as dataset_json_file:
            json.dump(self.learneble_data, dataset_json_file, indent=4)


if __name__ == '__main__':
    dt = Dataset()
    print(dt.learneble_data[0])
