import os
import sys
import pickle
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, 'src')

# Удаление всех существующих путей и добавление пути с src
sys.path = [src_path] + [p for p in sys.path if p != src_path]


from keras._tf_keras.keras.optimizers import Adam, AdamW
from keras._tf_keras.keras.losses import SparseCategoricalCrossentropy
from keras._tf_keras.keras.metrics import SparseCategoricalAccuracy, SparseTopKCategoricalAccuracy
import keras
import numpy as np
import tensorflow as tf

# from Model import GPT
# from Training import GPTTrainer
# from Data import Dataset, GPTTokenizer
from load_tokenizer import load_tokenizer
import predict
from src.Model import GPT
from src.Training import GPTTrainer
from src.Data import Dataset, GPTTokenizer

# model = GPT('model_test', 512, 10, 100)


class GPTTrain():
    def __init__(self, vocab_size, seq_length=50):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
    
    def load_model(self, path_to_model):
        self.model: GPT = keras.models.load_model(path_to_model)


vocab_size = 30002
seq_lenght = 50

def main():

    tokenizer = GPTTokenizer()
    dataset = Dataset()
    # model = keras._tf_keras.keras.models.load_model("./models/model_v_alpha_1.3.11761982137.2486532.keras1762001272.5561724.keras")
    model: GPT = keras._tf_keras.keras.models.load_model("./models/model_v_alpha_1.3.1rem_v2.11762175482.564715.keras1762176477.7573154.keras")
    model.name_of_model += "rem_v2.1"

    optimizer = AdamW(
        learning_rate=1.5e-5,
        weight_decay=0.001,  # 
        beta_1=0.9,
        clipnorm=1.0,
        beta_2=0.98
        )
    # loss_fn = SparseCategoricalCrossentropy(
    #     from_logits=True,
    #     label_smoothing=0.1  # значение от 0.05 до 0.2
    #     )

    model.compile(
        optimizer=optimizer,
        # loss=loss_fn, 
        loss="sparse_categorical_crossentropy", 
        # loss="sparse_categorical_crossentropy", 
        metrics=[ 
            SparseCategoricalAccuracy(name='accuracy'), 
            SparseTopKCategoricalAccuracy(k=5, name='top5_accuracy'), 
            SparseTopKCategoricalAccuracy(k=10, name='top10_accuracy'),
            ],
        jit_compile=True
        )
            # 'accuracy', 'perplexity', 
    model.get_config()
    trainer = GPTTrainer(model, dataset, tokenizer, 0.1)
    # trainer.prepare_dataset_to_tf(1000)
    # trainer.
    # predict.main(model)
    print(trainer.fit())
    predict.main(model)


if __name__ == '__main__':
    main()
