'''Usage : python train.py --dataset [data] --save-path [save dir]'''

import pandas as pd
import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tqdm import main
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow import keras
import keras_tuner as kt
import os
import argparse

class DataLoader(object):

    def __init__(self, path, ratio=(0.8,0.1,0.1)):
        X,y = self.load_dataset(path)
        self.split_dataset(X,y,ratio)
        self.X = X
        self.y = y


    def load_dataset(self, path):
        df = pd.read_csv(path, encoding='utf-8')
        X = df.drop(['Correct Entropy','Approx Entropy'], axis = 1)
        y = df['Correct Entropy']
        return X.to_numpy(), y.to_numpy()
        

    def split_dataset(self, X, y, ratio):
        '''
        Args:
            X: the training inputs array
            y: the ground truth data array
            ratio: (train, validation, test)
        '''
        assert sum(ratio) == 1
        seed = 42
        X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=ratio[2], random_state = seed)
        X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=ratio[1]/(ratio[0]+ratio[1]), random_state=seed) 

        self.X_train_full = X_train_full
        self.y_train_full = y_train_full
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test


    def __len__(self):
        return len(self.X) 


def hypermodel_1(hp):
    '''
    Args: hp is the hyperparameter in keras_tuner
    '''
    units = hp.Int(name="units", min_value=16, max_value=64, step=16)
    model = keras.Sequential([
        layers.Dense(units, activation="relu"),
        layers.Dense(1)
    ])
    optimizer = hp.Choice(name="optimizer", values=["rmsprop", "adam"])
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model 


def hypermodel_2(hp):
    return NotImplemented


def train_with_tuner(hypermodel, save_path, dataloader):

    tuner = kt.BayesianOptimization(hypermodel, 
                                    objective="val_loss", 
                                    max_trials=1, 
                                    executions_per_trial=2, 
                                    directory=save_path, 
                                    overwrite=True, 
                                    ) 

    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=5),
                keras.callbacks.TensorBoard(log_dir=save_path)] 

    tuner.search(x = dataloader.X_train, y = dataloader.y_train,\
                batch_size=128, epochs=1, validation_data=(dataloader.X_val, dataloader.y_val), 
                callbacks=callbacks, verbose=2,)

    top_n = 3
    best_hps = tuner.get_best_hyperparameters(top_n) 
    model = hypermodel(best_hps[0])

    return model



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    args = parser.parse_args()
    
    dataloader = DataLoader(args.dataset, (0.8, 0.1, 0.1))
    model = train_with_tuner(hypermodel_1, args.save_path, dataloader)
    # fine tune the model
    history = model.fit(dataloader.X_train_full, dataloader.y_train_full, 
                        epochs=5,
                        callbacks=keras.callbacks.TensorBoard(
                            log_dir=args.save_path,
                            write_graph=True))

    # save model history
    model.save(os.path.join(args.save_path, 'final_model'))
    hist_df = pd.DataFrame(history.history)
    with open(os.path.join(args.save_path, 'history.json'), 'w') as f:
        hist_df.to_csv(f)