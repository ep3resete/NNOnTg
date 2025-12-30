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

from Model import GPT
from Training import GPTTrainer
from Inference import GPTPredictor
from Data import Dataset, BPETokenizer
from Utils import LoadSetup

import keras
import numpy as np

def main(model=None, tokenizer=None, dataset=None):
    ls = LoadSetup()
    # tokenizer = BPETokenizer()
    model = ls.model
    dataset = ls.dataset
    tokenizer = ls.tokenizer

    model.get_config()
    model.summary()
    a = model(tf.constant([[3, 11, 2]]))
    print(tf.argmax(a[0]))
    predictor = GPTPredictor(model, tokenizer, False)
   
    prompt = ''
    while prompt != 'вых':
        prompt = input("Введите сообщение: ")
        if prompt == 'вых':
            break
        elif prompt == 'рес':
            predictor = GPTPredictor(model, tokenizer, False)
        else:
            # gpt_answer = predictor.generate(prompt, max_tokens=15, strategy='top_k')
            gpt_answer = predictor.generate(prompt, max_tokens=50, strategy='greedy')
        # print(f"Сообщение пользователя: {prompt}")
        print(gpt_answer)
    # print(trainer.fit())

if __name__ == '__main__':
    main()
               