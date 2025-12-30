import sentencepiece as spm
import json
from collections import defaultdict

class BPETokenizer(spm.SentencePieceProcessor):
    def __init__(self, vocab_size=None, learn=False):
        super().__init__()
        
        with open("config/PathConfig.json", 'r', encoding='utf-8') as path_config_file:
            paths = json.load(path_config_file)
            path_to_data_file_config = paths["DataConfig"]

        with open(path_to_data_file_config, 'r', encoding='utf-8') as config_data_file:
            data_config = json.load(config_data_file)
            self.vocab_size = vocab_size or data_config["vocab_size"]
            self.path_to_file_with_raw_dialogs = data_config["path_to_row_dialogs_data"]
            self.path_to_file_with_raw_texts = data_config["path_to_row_texts_data"]

        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<eos>': 2,
            '<user>': 3, 
            '<bot>': 4
        }
        
        self.learn = learn
        self.vocab = {}
        self.merges = {}  # Заглушка для совместимости
        self.idx_to_token = {}
        
        if learn:
            self.train_bpe()
        else:
            self.load_tokenizer("bpe_tokenizer.json")

    def open_file(self, path: str):
        with open(path, encoding='utf-8') as file_with_data:
            return json.load(file_with_data)

    def get_training_data(self):
        """Собирает все тексты для обучения"""
        all_texts = []
        
        # Загрука данных только когда нужно
        file_with_dialogs = self.open_file(self.path_to_file_with_raw_dialogs)
        file_with_texts = self.open_file(self.path_to_file_with_raw_texts)
        
        for dialog in file_with_dialogs:
            for phrase in dialog['dialog']:
                all_texts.append(phrase.lower())
        
        for text_obj in file_with_texts:
            all_texts.append(text_obj['text'].lower())
            
        return all_texts

    def train_bpe(self):
        """Обучает SentencePiece токенизатор"""
        print("Начало обучения BPE токенизатора...")
        
        all_texts = self.get_training_data()
        
        # Сохранение текстов во временный файл
        with open('temp_training_texts.txt', 'w', encoding='utf-8') as f:
            for text in all_texts:
                f.write(text + '\n')
        
        # Обучение SentencePiece
        spm.SentencePieceTrainer.train(
            input='temp_training_texts.txt',
            model_prefix='russian_bpe',
            vocab_size=self.vocab_size,
            character_coverage=0.9995,
            model_type='bpe',
            pad_id=0,
            unk_id=1,
            bos_id=2, 
            eos_id=3,
            user_defined_symbols=['<user>', '<bot>', '<eos>']
        )
        
        # Загрузка обученной модели в родительский класс
        self.load('russian_bpe.model')
        self._build_compatible_vocab()
        
        print(f"Обучение завершено. Размер словаря: {len(self.vocab)}")

    def _build_compatible_vocab(self):
        """Строим совместимый словарь"""
        self.vocab = {}
        
        # Добавление специальных токенов
        for token, idx in self.special_tokens.items():
            self.vocab[token] = idx
        
        # Добавление токенов из SentencePiece
        for i in range(self.get_piece_size()):
            token = self.id_to_piece(i)
            if token not in self.special_tokens:
                self.vocab[token] = i + len(self.special_tokens)
        
        # Обратное отображение
        self.idx_to_token = {idx: token for token, idx in self.vocab.items()}

    def tokenize_text(self, text: str) -> list:
        """Токенизирует текст - основной метод"""
        return self.encode_as_ids(text)

    def tokenize_dialog(self, dialog: list[str]) -> list[list[int]]:
        """Токенизирует диалог"""
        result = []
        for phrase in dialog:
            token_ids = self.tokenize_text(phrase)
            result.append(token_ids)
        return result

    def detokenize_text(self, token_ids: list) -> str:
        """Детокенизирует список индексов обратно в текст"""
        # Фильтрация специальных токены которые не умеет обрабатывать SentencePiece
        filtered_ids = []
        for token_id in token_ids:
            if token_id in self.idx_to_token:
                token = self.idx_to_token[token_id]
                if token in self.special_tokens:
                    continue
                filtered_ids.append(token_id)
        
        return self.decode_ids(filtered_ids)

    def save_tokenizer(self, path: str):
        """Сохраняет информацию о токенизаторе"""
        tokenizer_data = {
            'model_path': 'russian_bpe.model',
            'special_tokens': self.special_tokens,
            'vocab_size': self.vocab_size
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)

    def load_tokenizer(self, path: str):
        """Загружает токенизатор"""
        with open(path, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)
        
        self.special_tokens = tokenizer_data['special_tokens']
        self.vocab_size = tokenizer_data['vocab_size']
        
        self.load(tokenizer_data['model_path'])
        self._build_compatible_vocab()

if __name__ == '__main__':
    # Тестирование
    tokenizer = BPETokenizer(learn=True)
    
    test_dialog = ["привет, как дела?", "да норм, а у тебя?"]
    tokenized = tokenizer.tokenize_dialog(test_dialog)
    print("Токенизированный диалог:", tokenized)
    
    for i, tokens in enumerate(tokenized):
        original = tokenizer.detokenize_text(tokens)
        print(f"Оригинал: {test_dialog[i]}")
        print(f"Восстановлено: {original}")
        print("Совпадает?", test_dialog[i].lower() == original.lower())
        print("---")
  