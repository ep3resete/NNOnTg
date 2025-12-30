import tensorflow as tf
import json
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from collections import OrderedDict, Counter, defaultdict
import numpy as np
import random


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
        # path_to_file_with_row_tokenized_data = data_config["path_to_tokenized_file"] # Путь до файла с сырыми токенизированными данными
        path_to_file_with_row_tokenized_dialogs = data_config["path_to_tokenized_dialogs_file"] # Путь до файла с сырыми токенизированными данными
        path_to_file_with_row_tokenized_texts = data_config["path_to_tokenized_texts_file"] # Путь до файла с сырыми токенизированными данными
        vocab_size = data_config["vocab_size"]
        dataset_config = data_config["dataset_config"] # Конфиг датасета
        path_to_dataset_json = dataset_config["path_to_dataset_json"] # Путь до файла датасета в json формате
        shuffle_buffer_size = dataset_config["shuffle_buffer_size"]
        batch_size = dataset_config["batch_size"]
        del data_config # Удаление лишней переменной

    def __init__(self, skip_frequent_tokens=False, word_counts=None, is_pretrained=False,
                 filter_min_freq=10, filter_max_freq=1000, 
                 filter_rare_prob=0.1, filter_freq_prob=0.3):
        self.is_pretrained = is_pretrained
        if is_pretrained:
            self.raw_data = self.open_file(self.path_to_file_with_row_tokenized_texts)
            print(f"Загружен датасет по пути **{self.path_to_file_with_row_tokenized_texts}**")
        else:
            self.raw_data = self.open_file(self.path_to_file_with_row_tokenized_dialogs)
            print(f"Загружен датасет по пути **{self.path_to_file_with_row_tokenized_dialogs}**")
        self.learneble_data = self.get_learnable_dataset(skip_frequent_tokens, word_counts)
        
        # Применяем фильтрацию если указаны параметры
        if (filter_min_freq is not None and filter_max_freq is not None) and skip_frequent_tokens:
            # self.filter_dataset_by_frequency(
            #     min_freq=1,
            #     max_freq=20,
            #     rare_token_prob=filter_rare_prob,
            #     frequent_token_prob=filter_freq_prob
            # )
            # self.anti_spam_filter(self.learneble_data[0], self.learneble_data[1])
            # self.diasable_eos_id(self.learneble_data[0], self.learneble_data[1])
            self.diasable_neuron_id(self.learneble_data[0], self.learneble_data[1], 1)
            self.diasable_neuron_id(self.learneble_data[0], self.learneble_data[1], 6)
            self.balance_frequent_tokens(self.learneble_data[0], self.learneble_data[1])
            # self.anti_spam_filter(self.learneble_data[0], self.learneble_data[1])

    def diasable_neuron_id(self, X, Y, id=1):
        filtered_X = []
        filtered_Y = []
        removed_count = 0
        original_size = len(X)

        # eos_token_id = 6
        
        for x, y in zip(X, Y):
            if y == id:
                removed_count += 1
                continue
            filtered_X.append(x)
            filtered_Y.append(y)
        
        new_size = len(filtered_X)
        removed_count = original_size - new_size
        print(f"🗑️ Удалено таргетов: {removed_count}")
        print(f"Осталось примеров: {len(filtered_X)}")

        self.learneble_data = (filtered_X, filtered_Y)
        return filtered_X, filtered_Y
        

    def balance_frequent_tokens(self, X, Y, max_examples_per_token=100, max_frequency=100):
        """
        Балансирует датасет: для частых токенов оставляет только max_examples_per_token примеров
        """
        # special_tokens={'<eos>', "привет"}
        special_tokens={}
        # from collections import Counter, 
        
        # Считаем статистику
        removed_count = 0
        token_counts = Counter(Y)
        
        print(f"Всего уникальных токенов: {len(token_counts)}")
        original_size = len(X)
        
        
        filtered_X, filtered_Y = [], []
        # Группируем примеры по целевым токенам
        token_to_examples = defaultdict(list)
        for x, y in zip(X, Y):
            token_to_examples[y].append((x, y))
        
        balanced_tokens = set()
        
        for token, examples in token_to_examples.items():
            # Для специальных токенов - оставляем все
            # if token in special_tokens:
            #     filtered_X.extend([ex[0] for ex in examples])
            #     filtered_Y.extend([ex[1] for ex in examples])
            #     continue
                
            # Если токен встречается редко - оставляем все
            if len(examples) <= max_frequency:
                filtered_X.extend([ex[0] for ex in examples])
                filtered_Y.extend([ex[1] for ex in examples])
            else:
                # Для частых токенов - оставляем только max_examples_per_token случайных примеров
                selected_examples = random.sample(examples, min(max_examples_per_token, len(examples)))
                filtered_X.extend([ex[0] for ex in selected_examples])
                filtered_Y.extend([ex[1] for ex in selected_examples])
                balanced_tokens.add(token)
        
        # Статистика
        # return filtered_X, filtered_Y
        new_size = len(filtered_X)
        removed_count = original_size - new_size

        
        print(f"🔧 БАЛАНСИРОВКА ДАННЫХ:")
        print(f"Было примеров: {original_size}")
        print(f"Стало примеров: {new_size}")
        print(f"Удалено примеров: {removed_count}")
        print(f"Сбалансировано токенов: {len(balanced_tokens)}")
        print(f"Топ-10 сбалансированных токенов: {list(balanced_tokens)[:10]}")
        
        self.learneble_data = (filtered_X, filtered_Y)
        return filtered_X, filtered_Y

    def anti_spam_filter(self, X, Y, min_diversity=0.4):
        """
        Жесткая фильтрация против спама частыми токенами
        """
        # import numpy as np
        
        # Статистика по целевым токенам
        token_counts = Counter(Y)
        # total_samples = len(Y)
        
        # Топ-N самых частых токенов (спам-кандидаты)
        top_tokens = [token for token, count in token_counts.most_common()]
        
        filtered_X, filtered_Y = [], []
        spam_count = 0
        
        for x, y in zip(X, Y):
            # Правило 1: Удалить примеры где целевой токен в топ-20 частых
            if y in top_tokens[:20]:
                spam_count += 1
                continue
                
            # Правило 2: Удалить примеры с низким разнообразием в контексте
            unique_ratio = len(set(x)) / len(x) if len(x) > 0 else 0
            if unique_ratio < min_diversity:
                spam_count += 1
                continue
                
            # Правило 3: Удалить слишком короткие последовательности
            if len(x) < 4:
                spam_count += 1
                continue
                
            filtered_X.append(x)
            filtered_Y.append(y)
        
        print(f"Удалено спам-примеров: {spam_count}")
        print(f"Осталось примеров: {len(filtered_X)}")
        self.learneble_data = (filtered_X, filtered_Y)

        return filtered_X, filtered_Y


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
        
        # Применяем фильтрацию
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
            # 'remove' action - should_keep остается False
            
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
    
    @classmethod
    def open_file(self, path):
        """ Метод для открытия файла с сырми токенизированными данными """
        with open(path, 'r', encoding='utf-8') as file_with_row_data:
            row_data = json.load(file_with_row_data) # Сырые данные
            return row_data
    
    def prepare_dataset_with_padding(self, data, max_length=None, pad_token_id=0):
        """ Метод для форматирования данных из датасета """
        inputs, targets = data # Разложение всей переменной с данными на вход и таргет
        mli = max(inputs, key=len)
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
                # tf.cast(tf.convert_to_tensor(x, dtype=tf.int32), tf.int32),
                # tf.cast(tf.convert_to_tensor(y, dtype=tf.int32), tf.int32)
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

    def get_learnable_dataset(self, ignore_frequent_tokens=False, word_counts: OrderedDict=None):
        """ Метод для создания датасет из сырых токеннизированных данных
        Args:
            ignore_frequent_tokens: bool - Флаг, указывающий на то, нужно ли пропускать самые частые токены или нет
            is_pre_train_data: bool - 
            word_counts: OrderesDict - 
        """
        X = []  # Входы
        Y = []  # Выходы (цели)
        # Проверка на тип данных для обучения
        if self.is_pretrained: # Претреининг обучение
            # Перебор всех текстов для обучения
            for text in self.raw_data: 
                text = text["text"] # Текущий текст
                current_sequence = [] # Текущая длина. Нужно чтобы добавлять данные "лесенкой"

                for token in text:
                    # Проверка на то, есть ли сейчас какие-то данные в текущей последовательности токенов
                    if len(current_sequence) < 4 - 1: # Если последовательность пустая
                    # if len(current_sequence) < self.seq_length - 1: # Если последовательность пустая
                        # Добавление первого токена в общую последовательность
                        current_sequence.append(token) 
                    else: # Когда последовательность размера окна
                    # else: # Когда последовательность не пустая
                        # Проверка на то, является ли длина последовательности максимальной
                        if len(current_sequence) == self.seq_length: # Если является - то удаляются первые токены
                            current_sequence.pop(0) 
                        # Вход: текущая последовательность
                        X.append(current_sequence.copy())
                        # Выход: следующий токен
                        Y.append(token)
                        # Добавление токена к последовательности
                        current_sequence.append(token)
        else:
            # Перебор всех диалогов
            for dialog in self.raw_data:
                dialog = dialog['dialog'] # Текущий диалог
                current_sequence = [] # Текущая длина. Нужно чтобы добавлять данные "лесенкой"
                
                for j, message in enumerate(dialog):
                # for j, message in enumerate(dialog[:2]):
                    if j % 2 == 0:  
                        # Пользователь - добавление всей фразы
                        current_sequence.extend(message)
                    else:  
                        # Бот - добавление по токенам
                        for token in message:
                        # for token in message[:4]:
                            # if not (ignore_frequent_tokens and (word_counts[token] <= 1500) and (word_counts[token] >= 200)and token != 1):
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
            # self.create_tf_dataset(self.learneble_data, shuffle) # Данные для обучениия становятся в виде tf

        # self.create_tf_dataset(self.learneble_padding_data, shuffle) # Данные для обучениия становятся в виде tf
        # self.create_batches(self.batch_size) # Сбор датасета в бэтчи
        
        return self.learneble_data_tf

    def save_dataset_to_json(self):
        """ Метод сохраняет значения из self.learneble_data в файл dataset.json """
        with open(self.path_to_dataset_json, 'w', encoding='utf-8') as dataset_json_file:
            json.dump(self.learneble_data, dataset_json_file, indent=4)


if __name__ == '__main__':
    dt = Dataset(is_pretrained=True)
    print(dt.learneble_data[0])
