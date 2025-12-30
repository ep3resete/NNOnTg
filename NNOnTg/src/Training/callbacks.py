import json
import time
import pprint
import tensorflow as tf
import numpy as np
from keras._tf_keras.keras.callbacks import ModelCheckpoint
from keras.src.utils import file_utils
from Inference import GPTPredictor
from collections import Counter

class LoggingModelCheckpoint(ModelCheckpoint):
    """ Класс, которой унаследуется от ModelCheckpoint. Делает всё то же самое, но ещё и логгирует сохранение в файл"""
    def check_model(self, prompts):
        for prompt in prompts:
            predictor = GPTPredictor(self.model, self.tokenizer)
            predictor.generate(prompt, 20, self.strategy)
        update_dropout = False
        if update_dropout:
            self.model.update_dropout_rate()

    def __init__(self, tokenizer, filepath, model_name, path_to_log_json, monitor="val_loss", verbose=1, save_best_only=False, save_weights_only=False, mode="auto", save_freq="epoch", initial_value_threshold=None):
        filepath = filepath + str(time.time()) + ".keras"
        # self.strategy = 'top_k'
        self.strategy = 'greedy'
        # filepath = filepath + ".weights.h5"
        self.folder_path = filepath
        super().__init__(filepath, monitor, verbose, save_best_only, save_weights_only, mode, save_freq, initial_value_threshold)
        self.tokenizer = tokenizer
        self.path_to_log_json = path_to_log_json
        self.model_name = model_name
        self.check = False
        self.save = True

    def on_epoch_end(self, epoch, logs=None):
        """ Вызывается в конце каждой эпохи. 
        Сначала происходит сохранение модели (родительский метод), потом происходит логгирование
        """
        self.filepath = file_utils.path_to_string(self.folder_path + str(time.time()) + ".keras")
        self.filepath = self.filepath
        if self.save:
            super().on_epoch_end(epoch, logs)
        self._log_metrics(logs)

        test_prompts = [
            # Базовые
            "Привет",
            "Как дела?",
            "Что нового?",
            "Какой сегодня день?",
            "Как тебя зовут?",
            "планирую купить велосипед",
            "привет, как ты?",
            "завтра экзамен...",
            "привет, как ты? что поделываешь?",

            
            # Диалоговые  
            "Приготовил домашнюю пиццу - получилось даже лучше, чем в некоторых кафе",
            "Сегодня отлично погулял в парке, видел белок и певших птиц",
            "У меня немного болит голова, посоветуй что делать",
            "Как думаешь, стоит ли учить машинное обучение в 2024?",
            "Расскажи короткую шутку про программистов",
            
            # Сложные/творческие
            "Опиши ощущения человека, который впервые видит океан",
            "Если бы у тебя была суперсила, какую бы выбрал и почему?",
            "Что такое счастье в твоём понимании?",
            "Напиши короткое стихотворение о закате",
            "Представь, что ты можешь изменить одну вещь в мире - что выберешь?",
            
            # Контекстные
            "Вчера я рассказывал про свою поездку в горы. Как думаешь, стоит ли мне повторить её в этом году?",
            "Помнишь, я говорил про проблему с проектом? Так вот, сегодня ситуация ухудшилась...",
            
            # Стресс-тесты
            "Абракадабра непонятный бред текст",
            "",  # пустой промпт
            "1111111111111111111111111",
            "What is your name?",
            "Очень-очень-очень длинное предложение " * 10
        ]
        if self.check:
            self.check_model(test_prompts)
    
    def _log_metrics(self, logs):
        with open(self.path_to_log_json, 'r', encoding='utf-8') as json_file_with_logs:
            all_logs: dict = json.load(json_file_with_logs)
        
        # Проверка на то, существуют ли логи для сохраняемой модели
        if self.model_name not in all_logs.keys():
            # Если нет, то создаются такие логи
            all_logs[self.model_name] = {}
        epoch_key = f"epoch_{self._current_epoch + 1}"

        epoch_metrics = {}    
        for key, value in logs.items():
            if hasattr(value, 'numpy'):
                epoch_metrics[key] = float(value.numpy())
            else:
                epoch_metrics[key] = float(value)
        model_path = self.filepath.format(epoch=self._current_epoch + 1, **logs)
        epoch_metrics['model_path'] = model_path

        all_logs[self.model_name][epoch_key] = epoch_metrics

        # Открытие файла с логами
        with open(self.path_to_log_json, 'w', encoding='utf-8') as json_file_with_logs:
            # Запись логов в файл
            json.dump(all_logs, json_file_with_logs, indent=4)

class PredictionAnalyzer(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, tokenizer, num_samples=10):
        super().__init__()
        self.validation_data = validation_data
        self.tokenizer = tokenizer
        self.num_samples = num_samples
    
    def compare_inference_modes(self):
        """Сравниваем инференс из датасета и ручной инференс"""
        pass
            
    def _simple_autoregressive_generate(self, initial_tokens, max_length=20):
        """Простая авторегрессивная генерация"""
        generated = initial_tokens.copy()
        
        for step in range(max_length):
            input_tensor = tf.constant([generated], dtype=tf.int32)
            
            logits = self.model(input_tensor, training=False)
            
            # Определение позиции для генерации
            real_tokens = [t for t in generated if t != 0]
            current_pos = len(real_tokens) - 1
            
            # Логиты для текущей позиции
            if len(logits.shape) == 3:
                current_logits = logits[0, current_pos, :]
            else:
                current_logits = logits[0, :]
            
            # Greedy выбор
            next_token = tf.argmax(current_logits).numpy()
            
            generated.append(next_token)
            if len(generated) > 256:
                generated.pop(0)
            
            # Проверяем конец генерации
            eos_token = self.tokenizer.special_tokens["<eos>"]
            if next_token == eos_token + 4:
                break
        
        return generated

    def on_epoch_end(self, epoch, logs=None):
        print(f"\n┌{'─'*60}┐")
        print(f"│ Анализ эпохи {epoch+1:>2} │ Val Accuracy: {logs.get('val_accuracy', 0):.3f} │ Loss: {logs.get('val_loss', 0):.3f} │")
        print(f"└{'─'*60}┘")
        
        try:
            all_predictions = []
            correct_predictions = 0
            total_predictions = 0
            token_2325_count = 0
            
            for inputs, targets in self.validation_data.take(1):
                predictions = self.model(inputs)
                predicted_tokens = tf.argmax(predictions, axis=-1)
                
                batch_size = 256
                self.compare_inference_modes()
                
                all_targets = []
                for i in range(batch_size):
                    # Последовательности
                    input_seq = inputs[i].numpy().flatten().tolist()
                    target_seq = targets[i].numpy().flatten().tolist()
                    pred_seq = predicted_tokens[i].numpy().flatten().tolist()
                    
                    # Проверка на то, являются ли они списками
                    if not isinstance(target_seq, list):
                        target_seq = [target_seq]
                    if not isinstance(pred_seq, list):
                        pred_seq = [pred_seq]
                    
                    # Декодировка
                    input_text = self.tokenizer.decode(input_seq)
                    target_text = self.tokenizer.decode(target_seq)
                    pred_text = self.tokenizer.decode(pred_seq)
                    
                    # Статистика 
                    correct = sum(1 for p, t in zip(pred_seq, target_seq) if p == t)
                    total = len(target_seq)
                    accuracy = correct / total if total > 0 else 0
                    
                    correct_predictions += correct
                    total_predictions += total
                    token_2325_count += pred_seq.count(2325)
                    
                    # Форматирование вывода
                    input_short = input_text[:500] + "..." if len(input_text) > 500 else input_text
                    target_short = target_text[:200] + "..." if len(target_text) > 200 else target_text
                    pred_short = pred_text[:200] + "..." if len(pred_text) > 200 else pred_text
                    status = "✓" if pred_text == target_text else "✗"
                    if pred_text == target_text:
                        all_targets.append(pred_text)
                    
                    print(f"{status} {i+1:2d}. Вход: {input_short}")
                    print(f"     Цель: {target_short:<23} │ Предсказание: {pred_short}")
                    print(f"     Точность: {correct}/{total} ({accuracy:.1%})")
                    
                    # Если предсказание неверное, поиск разницы
                    if pred_text != target_text:
                        # Найдем первой позиции с ошибкой
                        for pos, (p, t) in enumerate(zip(pred_seq, target_seq)):
                            if p != t:
                                try:
                                    p_text = self.tokenizer.decode([p])
                                    t_text = self.tokenizer.decode([t])
                                except:
                                    p_text, t_text = f"[{p}]", f"[{t}]"
                                print(f"     Первая ошибка: поз.{pos} '{p_text}' вместо '{t_text}'")
                                break
                    
                    all_predictions.extend(pred_seq)
                    
                    if i < batch_size - 1:
                        print("     " + "─" * 50)
                
                break  # только один батч
            
            test_prompts = [
                "Привет",
                "Как дела?",
                "Что нового?",
                "Какой сегодня день?",
                "Как тебя зовут?",
                "Расскажи шутку",
                "Что такое искусственный интеллект?",
                "Помоги мне с советом"
            ]
            
            custom_generated_tokens = []
            
            for i, user_prompt in enumerate(test_prompts):
                try:
                    # Форматирование промпта как в диалоге
                    full_prompt = f"<user>{user_prompt}<eos><bot>"
                    tokens = self.tokenizer.tokenize_text(full_prompt)
                    tokens = [x for x in tokens if x != 9936]
                    
                    # Паддинг до 96
                    padded_tokens = [0] * (96 - len(tokens)) + tokens
                    
                    # Генерирация авторегрессивна
                    generated = self._simple_autoregressive_generate(padded_tokens, max_length=15)
                    
                    # Удаление паддинга и исходного промпта из результата
                    clean_result = [t for t in generated if t != 0]
                    response_tokens = clean_result[len(tokens):] if len(clean_result) > len(tokens) else []
                    
                    response_text = self.tokenizer.detokenize_text(response_tokens) if response_tokens else "[пусто]"
                    
                    print(f"{i+1:2d}. Промпт: '{user_prompt}'")
                    print(f"     Ответ: {response_text}")
                    print(f"     Токены: {response_tokens[:10]}{'...' if len(response_tokens) > 10 else ''}")
                    
                    custom_generated_tokens.extend(response_tokens)
                    
                    if i < len(test_prompts) - 1:
                        print("     " + "─" * 40)
                        
                except Exception as e:
                    print(f"Ошибка с промптом '{user_prompt}': {e}")
        

            # Общая статистика
            print(f"\n{'='*60}")
            print("  ОБЩАЯ СТАТИСТИКА:")
            print(f"   Всего токенов: {total_predictions}")
            print(f"   Верных предсказаний: {correct_predictions} ({correct_predictions/total_predictions*100:.1f}%)")
            print(f"   Токен 2325 встречается: {token_2325_count} раз ({token_2325_count/total_predictions*100:.1f}%)")
            print(f"   Самые частые из верно предсказанных токенов: {Counter(all_targets)[:10]}")
            # Топ токенов
            if all_predictions:
                token_counts = Counter(all_predictions)
                print(f"\nТОП-5 предсказанных токенов:")
                for token, count in token_counts.most_common(5):
                    try:
                        token_text = self.tokenizer.decode([token])
                        if len(token_text) > 10:
                            token_text = token_text[:10] + "..."
                    except:
                        token_text = f"[токен {token}]"
                    percentage = count / len(all_predictions) * 100
                    print(f"   '{token_text}': {count} раз ({percentage:.1f}%)")
            
            # print(f"{'='*60}\n")
            if custom_generated_tokens:
                custom_counts = Counter(custom_generated_tokens)
                print(f"\nТОП-5 сгенерированных токенов (кастом):")
                for token, count in custom_counts.most_common(5):
                    try:
                        token_text = self.tokenizer.decode([token])
                        if len(token_text) > 10:
                            token_text = token_text[:10] + "..."
                    except:
                        token_text = f"[токен {token}]"
                    percentage = count / len(custom_generated_tokens) * 100
                    print(f"   '{token_text}': {count} раз ({percentage:.1f}%)")
            
            print(f"{'='*60}\n")
            
                
        except Exception as e:
            print(f"Ошибка анализа: {e}")
