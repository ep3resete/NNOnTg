import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, 'src')

# Удаление всех существующих путей и добавление пути с src
sys.path = [src_path] + [p for p in sys.path if p != src_path]


from Data.tokenizer import BPETokenizer
import json


class TokenizereForDataset():
    def __init__(self, is_pretrained=False):
        """ is_pretrained: bool - Флаг для указания, какой тип датасета использовать. Претреининг или файн-тюнинг"""
        self.is_pretrained = is_pretrained
        self.tokenizer = BPETokenizer(learn=True)
        # self.tokenizer.load_tokenizer("./data/tokenizers/bpe_tokenizer1.json")
    
    def open_file_with_data(self, path):
        with open(path, encoding='utf-8') as file_with_dialogs:
            self.data = json.load(file_with_dialogs)

    def tokenize_file(self, path_to_row_data):
        self.open_file_with_data(path_to_row_data)
        self.res = [] # Переменная для результата преобразования исходного файла. Будет всё то же самое, что и в изначальном файле, нго с токенами
        # Проверка на то, собирается претреининг датасет или файн-тюнинг
        if self.is_pretrained: # Претрейнинг
            # Перебор всех текстов из файлов
            for text in self.data:
                text = text["text"] # Текст
                token_text = self.tokenizer.tokenize_text(text) # Преобразование текста в токены
                dct_text = {'text': token_text} # Обратное преобразование текста в словарь
                self.res.append(dct_text) # Добавление текста в переменную результатов
        else: # Файн-тюнинг
            # Перебор диалогов из файла
            for dialog in self.data:
                dialog = dialog["dialog"] # Диалог
                token_dialog = self.tokenizer.tokenize_dialog(dialog) # Преобразование диалога в токены
                dct_dialogs = {'dialog': token_dialog} # Обратное преобразование диалога в словарь
                self.res.append(dct_dialogs) # Добавление этого словаря в переменную результатов
    
    def create_tokenized_file(self, path_tokenized_file):
        """ Сохранение результатов токенезации в файл """
        with open(path_tokenized_file, 'w', encoding='utf-8') as file_with_dialog_tokens:
            json.dump(self.res, file_with_dialog_tokens, ensure_ascii=False, indent=4)
            print(f"Датасет сохранён по пути {path_tokenized_file}")
            # json.dump(self.res, file_with_dialog_tokens, ensure_ascii=False, indent=4, separators=(',', ': '))


def main():
    is_pretrained = 1
    with open("./config/DataConfig.json", 'r', encoding='utf-8') as config_data_file:
        config_data = json.load(config_data_file)
        if is_pretrained:
            path_to_row_data = config_data["path_to_row_texts_data"]
            path_to_tokenized_data = config_data["path_to_tokenized_texts_file"]
        else:
            path_to_row_data = config_data["path_to_row_dialogs_data"]
            print(path_to_row_data)
            path_to_tokenized_data = config_data["path_to_tokenized_dialogs_file"]
        

    tfd = TokenizereForDataset(is_pretrained)
    tfd.tokenize_file(path_to_row_data)
    tfd.create_tokenized_file(path_to_tokenized_data)
    print(len(tfd.tokenizer.vocab) + 1)

if __name__ == '__main__':
    main()

