#!/usr/bin/env python
# coding: utf-8

# ### The following provides the source code for reproducing results in section 3 of arXiv: 2305.00997 of using classical deep neural networks to predict von Neumann entropy. This is based on TensorFlow-Keras. 

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots
import tensorflow as tf
import os
import json
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python import metrics
import keras_tuner as kt
from keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras import regularizers
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')

plt.style.reload_library()
plt.style.use(['science','no-latex'])
plt.rcParams['figure.dpi'] = 500


"Hyperparameters" 
# Please adjust the hyperparameters and the hypermodel_base function below.

# the following are hyperparameters for the KerasTuner
max_trials = 100
executions_per_trial = 2
patience = 8
search_epochs = 500

top_n = 5 # we pick the top_n models, note that this has to be at least the same as the max-trials

# the following are hyperparameters for retrained models
batch_size = 512
Retrain_training_times = 20 
retrain_patience = 8
retrain_max_epochs = 1000


def hypermodel_base(hp):
    #units = hp.Int(name="units", min_value=16, max_value=64, step=16)
    activation = hp.Fixed("activation", "relu")
    #learning_rate = hp.Fixed("learning_rate", 9e-3)
    learning_rate = hp.Float("learning_rate", min_value=3e-3, max_value=9e-3, sampling="log")
    dropout_rate = hp.Float("dropout_rate", min_value=0.1, max_value=0.5, sampling="log")  # step=0.1
    initializer = tf.keras.initializers.GlorotNormal(seed=42) # GlorotNormal is the default one, we add a seed

    
    model = keras.Sequential()
    for i in range(hp.Int("num_layers", 1, 4)):
        if hp.Boolean("BatchNormalization"):  
            model.add(
            Dense(
                units=hp.Int(f"units_{i}", min_value=16, max_value=128, step=16), # num of units will be indep
                use_bias = False, # if we use BatchNormalization, set bias to false and no activation
                kernel_initializer=initializer,
                )
            )
            model.add(BatchNormalization())
            model.add(Activation(activation=activation)) # set activation after BatchNormalization
        else:
            model.add(
            Dense(
                units=hp.Int(f"units_{i}", min_value=16, max_value=128, step=16), # num of units will be indep
                activation=activation,
                kernel_initializer=initializer,
                )
            )
    if hp.Boolean("dropout"):
        model.add(layers.Dropout(rate=dropout_rate))

    model.add(Dense(1))
    
    optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=True)  #hp.Boolean("amsgrad")
    
    model.compile(optimizer=optimizer, loss="mse",
              metrics=["mae"])
    return model 



class DataLoader(object):
    def __init__(self, path, ratio=(0.8,0.1,0.1)): 
        X, y = self.load_dataset(path)
        self.split_dataset(X, y, ratio)
        self.X = X
        self.y = y
    
    def load_dataset(self, path): 
        df = pd.read_csv(path, encoding='utf-8')
        X = df.drop(['Correct Entropy','Approx Entropy'], axis = 1)
        y = df['Correct Entropy']
        return X, y

    def shuffle_dataset(self, path):
        df = pd.read_csv(path, encoding='utf-8')
        df_shuffle = df.sample(frac=1., axis=0, random_state = 42).reset_index(drop=True)
        df_shuffle1 = df_shuffle.drop(['Correct Entropy', 'Approx Entropy'], axis = 1)
        y_targets = df_shuffle['Correct Entropy']
        return df_shuffle1, y_targets

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
        X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, 
                                                          test_size=round(ratio[1]/(ratio[0]+ratio[1]), 16), 
                                                          random_state=seed) 

        self.X_train_full = X_train_full
        self.y_train_full = y_train_full
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test


def load_tuner(hypermodel, load_dir, load_project):
    tuner = kt.BayesianOptimization(hypermodel,
                    objective="val_loss", 
                    directory=load_dir, 
                    project_name=load_project,
                    overwrite=False,
                    max_trials=max_trials, 
                    executions_per_trial=executions_per_trial, 
                    )
    return tuner


def load_best_hyperparam(hypermodel, load_dir, load_project, top_n):
    tuner = load_tuner(hypermodel, load_dir, load_project)
    return tuner.get_best_hyperparameters(top_n)

  # if you want to see the top_n hyper params, use below
  # best_trials = tuner.oracle.get_best_trials(num_trials=top_n)
  # for trial in best_trials:
    # trial.summary()
    # model = tuner.load_model(trial)
    # Do some stuff to the model



def train_with_tuner_ensemble_models(data_path, hypermodel, hp_dir, hp_proj): 
    # the saved checkpoint is under "hp_dir/hp_proj/"

    dataloader = DataLoader(data_path)
    tuner = load_tuner(hypermodel, hp_dir, hp_proj)
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience)]
    #  keras.callbacks.ModelCheckpoint(model_name, save_best_only=True)] 
    tuner.search(x = dataloader.X_train, y = dataloader.y_train, 
                batch_size = batch_size, epochs=search_epochs, validation_data=(dataloader.X_val, dataloader.y_val), 
                callbacks=callbacks, verbose=1,)
    tuner.results_summary()
    return tuner


def train_with_tuner_ensemble_models_retrain(path_train, path_val, hypermodel, best_hps, max_epochs, model_save_path): 
    # Now we re-train each model with the best epoch*ratio and the full data.
    # We monitor loss with EarlyStopping, the epoch*ratio will be the maximum possible epoch.
    # We train each model multiple times and use the one with the best training loss.
    
    dataloader = DataLoader(path_train)
    dataloader_val = DataLoader(path_val)
    X_full, y_full = dataloader_val.shuffle_dataset(path_val)
    ratio = (len(dataloader.X_train)+len(dataloader.X_val))/len(dataloader.X_train)
    
    # best_hps = tuner.get_best_hyperparameters(top_n)

    callbacks = [keras.callbacks.EarlyStopping(monitor="loss", patience=retrain_patience, restore_best_weights=True)]
    
    model_retrain = {}
    model_history_retrain = {}
    
    df_store_test = {}
    df_store_test_final = {}
    df_store_val = {}
    df_store_test_over_training = {}
    df_store_val_over_training = {}
    
    relative_errors_sum = {}
    
    
    
    pred_df = pd.DataFrame(dataloader.y_test).reset_index(drop=True) # the targets of the original train_test split
    pred_val_df = pd.DataFrame(y_full) # the targets of the full validation dataset
    
    Average_Epoch_List = []
    
    model_re_save = {}
    model_number_save = []
    
    for i in range(0, top_n):   
#         n = 0
#         while n < training_times:
#             Average_Epoch_List.append(max_epochs)
#             Average_Epoch_List.append(Average_Epoch[str(i) + str(n)])
#             n = n + 1
        j = 0
        while j < Retrain_training_times:
#             best_epoch = int(np.array(Average_Epoch_List).mean())
#             Best_training_epochs = int(best_epoch * ratio)
            model_retrain[str(i)] = hypermodel(best_hps[i])
            model_re = model_retrain[str(i)]

            model_re.fit(dataloader.X_train_full, dataloader.y_train_full, 
                          epochs = max_epochs,
                          batch_size = batch_size,
                          callbacks=callbacks,
                          verbose=1,
                  )
            
            model_re_save[str(i) + str(j)] = model_re
            model_history_retrain[str(i) + str(j)] = model_re.history.history

            test_predictions = model_re.predict(dataloader.X_test) # for the original train-test split
            test_pred = pd.DataFrame(test_predictions)
            test_pred.columns = ['Model_' + str(i) + ' #'+ str(j) + ' Predictions']
            df_store_test[str(i) + str(j)] = test_pred
            
            # save final model
            model_re.save(os.path.join(model_save_path, f'top_{i}_times_{j}'))

            j = j + 1


        # We choose the smallest overall relative error in the "test dataset" as the final output
        df_test_all = df_store_test[str(i) + str(0)]
        for k in range(1, Retrain_training_times):
            df_test_all = pd.concat([df_test_all, df_store_test[str(i) + str(k)]], axis=1)
            
        df_compare = pd.concat([pred_df, df_test_all], axis = 1)
        for l in range(0, Retrain_training_times):
            df_rel_error = (abs(df_compare['Correct Entropy']-df_compare['Model_' + str(i) + ' #'+ str(l) + ' Predictions'])/df_compare['Correct Entropy'])*100
            df_compare.insert(2+2*l, str(i)+'Rel Error (%) ' + str(l), df_rel_error)
        print(df_compare)
        relative_errors_sum[str(i)] = df_compare[[str(i)+'Rel Error (%) '+str(m) for m in range(0, Retrain_training_times)]].sum()
        model_number = df_compare[[str(i)+'Rel Error (%) '+str(m) for m in range(0, Retrain_training_times)]].sum().argmin()
        model_number_save.append(model_number)
        print(f"{relative_errors_sum[str(i)]}")
        print(f"# {model_number} has the smallest relative errors.")
        df_store_test_final[str(i)] = df_compare['Model_' + str(i) + ' #'+ str(model_number) + ' Predictions']
        
        # Then we choose the final model to predict the PredVal dataset
        model_re = model_re_save[str(i)+str(model_number)]
        
        val_predictions = model_re.predict(X_full) # for the Pred_validation dataset
        val_pred = pd.DataFrame(val_predictions)
        val_pred.columns = ['Model_' + str(i) + ' Pred_Val']
        df_store_val[str(i)] = val_pred



        # save final model
        model_re.save(os.path.join(model_save_path, 'top_'+str(i)))
    
    relative_errors_sum_df = pd.DataFrame(relative_errors_sum[str(0)])
    for k in range(0, top_n):
        relative_errors_sum_df = pd.concat([relative_errors_sum_df, pd.DataFrame(relative_errors_sum[str(k)])], axis=0)
    print(f"# {relative_errors_sum_df.idxmin()} has the smallest relative errors.")
    filepath = os.path.join(model_save_path, 'model_history_retrain.json') 
    with open(filepath, 'w') as handle:
        json.dump(model_history_retrain, handle)
        

    model_numbers_save_str = [str(i) for i in model_number_save]
    path = os.path.join(*model_numbers_save_str)
    with open(os.path.join(model_save_path, "model_number.txt"), "w") as f:
        f.write(path)
    
     # this shows how to load a model
#     print(load_model(save_model_path+'/top_0').summary())
#     print(load_model(save_model_path+'/top_1').summary())

        
    return df_store_test_final, df_store_val, pred_df, pred_val_df, df_compare


def Ensemble_Retrained_Plots(save_model_path, save_path, best_model):
    
    # Load the model_history_retrain
    with open(os.path.join(save_model_path, "model_history_retrain.json"), "r") as f:
        model_history_retrain = json.load(f)

    # Load the model_number
    with open(os.path.join(save_model_path, "model_number.txt"), "r") as f:
        contents = f.read()
    contents = contents.split("\n")
    contents = [i.replace("\\", "") for i in contents]
    str_list = str(contents[0])
    model_number_save = [int(i) for i in str_list]
        
    x_range_list = []    
    legend_labels = []
    #fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=False, figsize=(20, 16))
    fig, (ax1, ax3) = plt.subplots(2, 1, sharex=False, figsize=(20, 16))
    y_max = []
    y_max_mae = []
    j = 0
    while j < top_n:
        y_max.append(max(model_history_retrain[str(j)+str(model_number_save[j])]["loss"][:].copy()))
        y_max_mae.append(max(model_history_retrain[str(j)+str(model_number_save[j])]["mae"][:].copy()))
        j = j +1

    i = 0
    while i < top_n:
        y_loss = model_history_retrain[str(i)+str(model_number_save[i])]['loss']
        y_metrics = model_history_retrain[str(i)+str(model_number_save[i])]["mae"]
        x = range(1,len(y_loss)+1)

        ax1.plot(x, y_loss, '.--')
        #ax2.plot(x, y_loss, '.--')

        ax3.plot(x, y_metrics, '.--')
        #ax4.plot(x, y_metrics, '.--')

        x_range_list.append(len(y_loss))
        
        legend_labels.append('Model_' + str(i) + ' (Epoch= '+ str(x_range_list[i])+ ')')
        
        i = i + 1
    
    y_loss = model_history_retrain[str(i)+str(model_number_save[i])]['loss']
    y_metrics = model_history_retrain[str(i)+str(model_number_save[i])]["mae"]
    x = range(1,len(y_loss)+1)

    ax1.plot(x, y_loss, '.--')
    #ax2.plot(x, y_loss, '.--')

    ax3.plot(x, y_metrics, '.--')
    #ax4.plot(x, y_metrics, '.--')

    x_range_list.append(len(y_loss))

    legend_labels.append('Model_' + str(i) + ' (Epoch= '+ str(x_range_list[i])+ ')')
        

    fig.legend(legend_labels, loc="right", bbox_to_anchor=(1.13, 0.5), frameon=False, edgecolor='black', borderpad=1, labelspacing=2, handlelength=3) 
    ax1.set_title("Loss Function")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("MSE")
    ax1.set_xlim([1, len(y_loss)-retrain_patience])
    ax1.set_xticks(range(50,max(x_range_list),50))
    ax1.set_yscale('log')
    #ax1.set_xticklabels(range(5,max(x_range_list),5))


    ax3.set_title("Metrics")
    ax3.set_xlabel("Epochs")
    ax3.set_ylabel("MAE")
    ax3.set_xlim([1, len(y_loss)-retrain_patience])
    ax3.set_xticks(range(50,max(x_range_list),50))
    #ax3.set_xticklabels(range(5,max(x_range_list),5))
    ax3.set_yscale('log')


    fig.tight_layout()
    plt.savefig(os.path.join(save_path, 'Ensem_Retrain.png'))
    
def generate_figures(train_data_path, test_data_path, model_path, save_fig_dir, best_model, best_model_number):
    
    i = best_model
    j = best_model_number
    
    train_dataloader = DataLoader(train_data_path)
    df_train = pd.DataFrame(train_dataloader.y_test)
    df_train
    df = pd.read_csv(test_data_path, encoding='utf-8')
    df_shuffle = df.sample(frac=1., axis=0, random_state = 42)
    X_test  = df_shuffle.drop(['Correct Entropy', 'Approx Entropy'], axis = 1)
    y_test = df_shuffle['Correct Entropy']
    df_test = pd.DataFrame(y_test)
    df_test
    
    ymin = pd.concat([df_train, df_test], axis = 1)[['Correct Entropy','Correct Entropy']].min().min()*0.90
    ymax = pd.concat([df_train, df_test], axis = 1)[['Correct Entropy','Correct Entropy']].max().max()*1.05

    def compute_model_prediction(X_test, y_test, model):
        inputs = X_test.to_numpy()
        actual = y_test.to_numpy()
        approx = inputs.sum(1)
        sort_idx = np.argsort(actual)
        inputs, actual, approx = inputs[sort_idx], actual[sort_idx], approx[sort_idx]
        pred = model.predict(inputs).reshape(-1)
        
        return (pred, actual, approx)


    def plot_entropy_comparison(pred, actual, approx, save_fig_dir):
        plt.figure(figsize=(10,10))
        plt.plot(actual)
        plt.plot(pred)
        plt.plot(approx)
        plt.legend(["von Neumann entropy", "Model predictions", "Approximate entropy"])
        plt.ylim([ymin, ymax])
        plt.savefig(save_fig_dir)
        plt.show()
    
    model = load_model(save_model_path+'/top_'+str(i)+'_times_'+str(j))
    
    # plot loss and metric
    # Load the model_history_retrain
    with open(os.path.join(save_model_path, "model_history_retrain.json"), "r") as f:
        model_history_retrain = json.load(f)

#     fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False, figsize=(20, 16))
    fig1, ax1 = plt.subplots(figsize=(15, 5))
#     fig2, ax2 = plt.subplots(figsize=(15, 12))
    y_loss = model_history_retrain[str(i)+str(j)]['loss']
    y_metrics = model_history_retrain[str(i)+str(j)]["mae"]
    x = range(1,len(y_loss)+1)

    ax1.plot(x, y_loss, '--', alpha=1.0, color=(1,0,0))
#     ax2.plot(x, y_metrics, '--', alpha=1.0, color=(1,0,0))
    
    ax1.set_title("Loss Function")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("MSE")
    ax1.set_xlim([1, len(y_loss)-retrain_patience])
    xticks = list(range(50,len(y_loss)-retrain_patience,50))
    xticks.append(len(y_loss)-retrain_patience)
    ax1.set_xticks(xticks)
    #ax1.set_xticks(range(50,len(y_loss)-retrain_patience,50))
    ax1.set_yscale('log')
    plt.savefig(os.path.join(save_fig_dir, 'Loss.jpg'))
    
#     ax2.set_title("Metric")
#     ax2.set_xlabel("Epochs")
#     ax2.set_ylabel("MAE")
#     ax2.set_xlim([1, len(y_loss)-retrain_patience])
#     ax2.set_xticks(xticks)
#     ax2.set_yscale('log')
#     fig.tight_layout()
#     plt.savefig(os.path.join(save_fig_dir, 'Ensem_Retrain.png'))
    
    # plot test dataset
    train_dataloader = DataLoader(train_data_path)
    X_test = train_dataloader.X_test
    y_test = train_dataloader.y_test
    train_pred = compute_model_prediction(X_test, y_test, model)
    plot_entropy_comparison(*train_pred, save_fig_dir+"/entropy_compare1.jpg")

    # plot unseen dataset
    df = pd.read_csv(test_data_path, encoding='utf-8')
    df_shuffle = df.sample(frac=1., axis=0, random_state = 42)
    X_test  = df_shuffle.drop(['Correct Entropy', 'Approx Entropy'], axis = 1)
    y_test = df_shuffle['Correct Entropy']
    test_pred = compute_model_prediction(X_test, y_test, model)
    plot_entropy_comparison(*test_pred, save_fig_dir+"/entropy_compare2.jpg")

    # relative error
    diff = train_pred[0]-train_pred[1]
    train_error = (np.abs(diff)/train_pred[1])*100.
    diff = test_pred[0]-test_pred[1]
    test_error = (np.abs(diff)/test_pred[1])*100.
    fig_error, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    ax1.plot(train_error, '.')
    ax2.plot(test_error, '.')
    ax1.set_title('Relative Errors (%) for Test Data')
    ax2.set_title("Relative Errors (%) for Unseen Data")
    plt.savefig(save_fig_dir+"/relative_error.jpg")
    plt.show()
    
    # dist plot
    fig_dist, ax3 = plt.subplots(1,1,figsize=(10,5))
    sns.distplot(train_error)
    sns.distplot(test_error)
    ax3.set_title('Density Plot of Relative Errors for Test Data')
    ax3.set_xlabel('Relative Errors (%)')
    ax3.set_ylabel('Density')
    plt.savefig(save_fig_dir+"/relative_error_density.jpg")

def Ensembel_Tables_Generation(path, path_val, df_store_test_final, df_store_val, pred_df, pred_val_df, save_path):
    
    df_store_test_final

    dataloader = DataLoader(path)

    dataloader_val = DataLoader(path_val)
    X_full, y_full = dataloader_val.shuffle_dataset(path_val)
    
    
    df_test_all = df_store_test_final[str(0)]
    df_val_all = df_store_val[str(0)]
#     for i in range(1, top_n):
#         df_test_all = pd.concat([df_test_all, df_store_test_final[str(i)]], axis=1)
#         df_val_all = pd.concat([df_val_all, df_store_val[str(i)]], axis=1)

#     df_test_all_mean = df_test_all.mean(axis=1)

#     df_compare = pd.concat([pred_df, df_test_all_mean, df_test_all], axis = 1)

    df_compare = pd.concat([pred_df, df_test_all], axis = 1)
    # Renaming the columns
    old_column_names = df_compare.columns.tolist()
    column_mapping = {old_column_names[0]: 'Correct Entropy', old_column_names[1]: 'Model Predictions'}
    df_compare = df_compare.rename(columns=column_mapping)
    
    df_rel_error = (abs(df_compare['Correct Entropy']-df_compare['Model Predictions'])/df_compare['Correct Entropy'])*100
    df_compare.insert(2, 'Relative Errors (%) for Models', df_rel_error)
    df_compare = df_compare.sort_values(by=['Correct Entropy']) # reorder the values
    print(df_compare)

#    df_val_all_mean = df_val_all.mean(axis=1)

#    df_compare_val = pd.concat([pred_val_df, df_val_all_mean, df_val_all], axis = 1)

    df_compare_val = pd.concat([pred_val_df, df_val_all], axis = 1)
    
    # Renaming the columns
    old_column_names = df_compare_val.columns.tolist()
    column_mapping = {old_column_names[0]: 'Correct Entropy', old_column_names[1]: 'Model Predictions'}
    df_compare_val = df_compare_val.rename(columns=column_mapping)
    
    df_rel_error_val = (abs(df_compare_val['Correct Entropy']-df_compare_val['Model Predictions'])/df_compare_val['Correct Entropy'])*100
    df_compare_val.insert(2, 'Relative Errors (%) for Models', df_rel_error_val)
    df_compare_val = df_compare_val.sort_values(by=['Correct Entropy']) # reorder the values
    print(df_compare_val)

    fig_error, (ax5, ax6) = plt.subplots(1, 2, figsize=(10,5))
    ax5.plot(np.arange(0, len(dataloader.X_test)), df_compare['Relative Errors (%) for Models'], '.')
    ax5.set_title("Relative Errors (%) for Test Data")

    ax6.plot(np.arange(0, len(X_full)), df_compare_val['Relative Errors (%) for Models'], '.')
    ax6.set_title("Relative Errors (%) for Unseen Data")

    fig_error.tight_layout()
    plt.savefig(os.path.join(save_path, 'Ensemble_Table_Gen.png'))
    return df_compare, df_compare_val

def Test_Set_Plot(path, df_compare, df_compare_val, save_path):
    # The plot will automatically determine the lower and upper ylim based on the smallest and largest values of the Test and PredVal Sets.
    # We need to input both df_compare and df_compare_val
    df = pd.read_csv(path, encoding='utf-8')
    X = df.drop(['Correct Entropy'], axis = 1)
    y = df['Correct Entropy']
    seed = 42
    ratio=(0.8,0.1,0.1)
    assert sum(ratio) == 1
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=ratio[2], random_state = seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, 
                                                          test_size=round(ratio[1]/(ratio[0]+ratio[1]), 16), 
                                                          random_state=seed) 
    X_test = X_test.reset_index(drop=True)

    df_compare_1 = pd.concat([df_compare, X_test], axis = 1)
    df_compare_2 = df_compare_1.sort_values(by='Correct Entropy', ascending=True).reset_index(drop=True)
    
    # Determine the ylim
    ylim=(pd.concat([df_compare, df_compare_val], axis = 1)[['Correct Entropy','Correct Entropy']].min().min()*0.95, pd.concat([df_compare, df_compare_val], axis = 1)[['Correct Entropy','Correct Entropy']].max().max()*1.05) 
    # We expand slightly the ylim.
    df_compare_2.plot(y=['Correct Entropy', 'Model Predictions', 'Approx Entropy'], use_index=True, figsize=(10, 10), ylim=ylim)
    plt.legend(["von Neumann Entropy", "Model Predictions", "Approximate Entropy"])
    plt.savefig(os.path.join(save_path, 'Test_Set_Plot.png'))

    return df_compare_2, ylim
    

def Pred_Val_Plot(path, df_compare_val, ylim, save_path):
    # The plot will have the same ylim as the Test_Set_Plot
    df = pd.read_csv(path, encoding='utf-8')
    df_shuffle = df.sample(frac=1., axis=0, random_state = 42).reset_index(drop=True)
    X = df_shuffle.drop(['Correct Entropy'], axis = 1)
    
    df_compare_1 = pd.concat([df_compare_val, X], axis = 1)
    df_compare_2 = df_compare_1.sort_values(by='Correct Entropy', ascending=True).reset_index(drop=True)
    df_compare_2.plot(y=['Correct Entropy', 'Model Predictions', 'Approx Entropy'], use_index=True, figsize=(10, 10), ylim=ylim)
    plt.legend(["von Neumann Entropy", "Model Predictions", "Approximate Entropy"])
    plt.savefig(os.path.join(save_path, 'Pred_Val_Plot.png'))

    return df_compare_2


# In[ ]:


# Please put the directories of the datasets and the save paths below

train_data_path = ''
test_data_path = ''
save_hp_dir = ''
save_hp_proj = ''
save_model_path = ''
save_fig_dir = ''

# this is to assure the path name is correct
import os
assert(os.path.exists(train_data_path))
assert(os.path.exists(test_data_path))
assert(os.path.exists(save_hp_dir))
assert(os.path.exists(save_model_path))
assert(os.path.exists(save_fig_dir))


# In[ ]:


# One could run the following line by line for each example considered in section 3 of arXiv:2305.00997

# The following initialize the KerasTuner with the specified hyperparameters range
tuner = train_with_tuner_ensemble_models(train_data_path, hypermodel_base, save_hp_dir, save_hp_proj)

# The following load the best configurations found by KerasTuner and retrain the models by also including the validation data
best_hps = load_best_hyperparam(hypermodel_base, save_hp_dir, save_hp_proj, top_n)
df_store_test_final, df_store_val, pred_df, pred_val_df, df_compare = train_with_tuner_ensemble_models_retrain(train_data_path, test_data_path, hypermodel_base, best_hps, retrain_max_epochs, save_model_path)

# The following generates the figures for the loss function, predictions, and the relative errors.
generate_figures(train_data_path, test_data_path, save_model_path, save_fig_dir, 2, 10)


# ### The following provides the source code for reproducing results in section 4 of arXiv: 2305.00997 of using treating the Renyi entropies as sequential deep learning. Again based on TensorFlow-Keras. 

# In[74]:


from tensorflow.python import metrics
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# import scienceplots
import tensorflow as tf
import re
import os
import json
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
from keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras import regularizers
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')

# plt.style.reload_library()
# plt.style.use(['science','no-latex'])

plt.rcParams['figure.dpi'] = 500

"Hyperparameters" 
# Please adjust the hyperparameters and the parameters in hyper_RNN_model below.

# The following are hyperparameters for the datasets
MIN = 5 # the length of each vector
k = 1000 # how many datasets we want to use out of the full 10000 sets
ratio_original = (0.6, 0.2, 0.2)
ratio = (0.8, 0.0, 0.2) 

# the following are hyperparameters for the KerasTuner (Due to small learning rate, we should increase search_epochs)
max_trials = 300
executions_per_trial = 2 # The final output will be the average of this.
patience = 8
search_epochs = 500

# the following are hyperparameters for finding the best epoch
best_epoch_patience = 15 # We use a large patience value
best_epoch_training_epochs = 300
best_epoch_batch_size = 2048

# We train the best model training_times, and select the best_N out of the training_times.
Best_training_epochs = 1500
retrain_patience = 10
training_times = 30
best_N = 7

# the following are hyperparameters for models
batch_size = 2048

def hyper_RNN_model(hp): 
    '''We should only consider stacking RNN layers if there are bottlencks in the performance.
    If we want to stack layers, use hp.Boolean in the block of each layer.'''
    # note return_sequences = True only when stacking multiple RNNs, the final layer cannot have it.
    
    units = hp.Int(name="units", min_value=64, max_value=256, step=16)
    units2 = hp.Int(name="units", min_value=32, max_value=128, step=8)
    #units3 = hp.Int(name="units", min_value=8, max_value=32, step=8)
    DenseUnits = hp.Int(name="units", min_value=16, max_value=32, step=8)
    activation = hp.Fixed("activation", "relu")
    #learning_rate = hp.Fixed("learning_rate", 5e-5)
    learning_rate = hp.Float("learning_rate", min_value=1e-5, max_value=1e-4, sampling="log")
    dropout_rate = hp.Float("dropout_rate", min_value=0.2, max_value=0.5, sampling="log")
    recurrent_dropout = hp.Float(name="recurrent_dropout", min_value=0.1, max_value=0.3, sampling="log")
    #initializer = tf.keras.initializers.GlorotUniform(seed=42) # GlorotNormal or GlorotUniform, we add a seed

    model = keras.Sequential()
    
    #model.add(Masking(mask_value=0.0, input_shape=(None, 1)))

    #layers = hp.Choice(name="layers", values=["One-RNN", "Two-RNN"])
    #recurrent_dropout = recurrent_dropout,
    # if layers == "One-RNN":
    #     model.add(SimpleRNN(units, activation = activation,  input_shape=(None, 1)))
    # else:
    #     model.add(SimpleRNN(units, return_sequences = True, 
    #                 activation = activation, recurrent_dropout = recurrent_dropout, input_shape=(None, 1)))
    #     model.add(SimpleRNN(units2, activation = activation, input_shape=(None, 1)))

    layers = hp.Choice(name="layers", values=["One-RNN with Dropout", "One-RNN without Dropout", "Two-RNN with Dropout", "Two-RNN without Dropout"])
    if layers == "One-RNN with Dropout":
        model.add(SimpleRNN(units, activation = activation, recurrent_dropout = recurrent_dropout, input_shape=(None, 1)))
        if hp.Boolean("LayerNormalization"):
            model.add(LayerNormalization())
    elif layers == "One-RNN without Dropout":
        model.add(SimpleRNN(units, activation = activation, input_shape=(None, 1)))
        if hp.Boolean("LayerNormalization"):
            model.add(LayerNormalization())
    elif layers == "Two-RNN with Dropout":
        model.add(SimpleRNN(units, return_sequences = True, 
                    activation = activation, recurrent_dropout = recurrent_dropout, input_shape=(None, 1)))
        if hp.Boolean("LayerNormalization"):
            model.add(LayerNormalization())
        model.add(SimpleRNN(units2, activation = activation, input_shape=(None, 1)))
    else:
        model.add(SimpleRNN(units, return_sequences = True, 
                    activation = activation, input_shape=(None, 1)))
        if hp.Boolean("LayerNormalization"):
            model.add(LayerNormalization())
        model.add(SimpleRNN(units2, activation = activation, input_shape=(None, 1)))

    # layers = hp.Choice(name="layers", values=["Two-RNN with Dropout", "Two-RNN without Dropout", "Three-RNN with Dropout", "Three-RNN without Dropout"])
    # if layers == "Two-RNN with Dropout":
    #     model.add(SimpleRNN(units, return_sequences = True, 
    #                 activation = activation, recurrent_dropout = recurrent_dropout, input_shape=(None, 1)))
    #     model.add(SimpleRNN(units2, activation = activation, input_shape=(None, 1)))
    # elif layers == "Two-RNN without Dropout":
    #     model.add(SimpleRNN(units, return_sequences = True, 
    #                 activation = activation, input_shape=(None, 1)))
    #     model.add(SimpleRNN(units2, activation = activation, input_shape=(None, 1)))
    # elif layers == "Three-RNN with Dropout":
    #     model.add(SimpleRNN(units, return_sequences = True, 
    #                 activation = activation, recurrent_dropout = recurrent_dropout, input_shape=(None, 1)))
    #     model.add(SimpleRNN(units2, return_sequences = True, 
    #                 activation = activation, recurrent_dropout = recurrent_dropout, input_shape=(None, 1)))
    #     model.add(SimpleRNN(units3, activation = activation, input_shape=(None, 1)))
    # else:
    #     model.add(SimpleRNN(units, return_sequences = True, 
    #                 activation = activation, input_shape=(None, 1)))
    #     model.add(SimpleRNN(units2, return_sequences = True, 
    #                 activation = activation, input_shape=(None, 1)))
    #     model.add(SimpleRNN(units3, activation = activation, input_shape=(None, 1)))
    if hp.Boolean("Dense"):
         model.add(Dense(DenseUnits))
    if hp.Boolean("dropout"):
        model.add(Dropout(rate=dropout_rate))
    model.add(Dense(1))

    optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=hp.Boolean("amsgrad"))

    model.compile(optimizer=optimizer, loss="mse",
              metrics=["mae"])
    return model

class DataLoader_Sequence(object):
    def __init__(self, path):
        df_shuffle = self.load_dataset_sequence(path)
    
    def load_dataset_sequence(self, path): 
        seed = 42
        df = pd.read_csv(path, encoding='utf-8').drop(['Correct Entropy','Approx Entropy'], axis = 1)
        df_shuffle = df.sample(frac=1., axis=0, random_state = 42).reset_index(drop=True)
        return df_shuffle
    
    def split_dataset_sequence_CNN(self, df_shuffle, MIN, k, ratio): # no zero padding
        MIN = MIN # the crucial difference with RNN, we only take the past MIN steps, this will also be the legth of vector
        k = k     # how many datasets we want to use
        ratio = ratio # the ratio of train-val-test split, the split is in "timesteps"
        full_data_transposed = df_shuffle.T
        
        num_train_samples = int(ratio[0] * len(full_data_transposed))
        num_val_samples = int(ratio[1] * len(full_data_transposed))
        num_test_samples = len(full_data_transposed) - num_val_samples - num_train_samples
        
        X_train_store = {}
        y_train_store = {}
        for j in range(1, k+1):
            train_data = full_data_transposed.iloc[range(0, num_train_samples), range(j-1, j)]
            X_train = np.zeros((num_train_samples-MIN, MIN, 1)) # length is MIN
            y_train = np.zeros((num_train_samples-MIN, 1))
            for i in range(0, num_train_samples-MIN):
                X_train[i, -(MIN):, :] = train_data[i:i+MIN] # the second argument means we take the last (i+MIN) values
                y_train[i, :] = train_data[i+MIN:i+MIN+1]        
            X_train_store["group" + str(j)] = X_train
            y_train_store["group" + str(j)] = y_train
        X_train_full = X_train_store['group' + str(1)]
        y_train_full = y_train_store['group' + str(1)]
        for j in range(2, k+1):
            X_train_full = np.append(X_train_full, X_train_store['group' + str(j)], axis=0)
            y_train_full = np.append(y_train_full, y_train_store['group' + str(j)], axis=0)
        
        X_val_store = {}
        y_val_store = {}
        for j in range(1, k+1):
            val_data = full_data_transposed.iloc[range(num_train_samples, num_train_samples+num_val_samples), range(j-1, j)]
            X_val = np.zeros((num_val_samples-MIN, MIN, 1))
            y_val = np.zeros((num_val_samples-MIN, 1))
            for i in range(0, num_val_samples-MIN):
                X_val[i, -(MIN):, :] = val_data[i:i+MIN] 
                y_val[i, :] = val_data[i+MIN:i+MIN+1]        
            X_val_store["group" + str(j)] = X_val
            y_val_store["group" + str(j)] = y_val
        X_val_full = X_val_store['group' + str(1)]
        y_val_full = y_val_store['group' + str(1)]
        for j in range(2, k+1):
            X_val_full = np.append(X_val_full, X_val_store['group' + str(j)], axis=0)
            y_val_full = np.append(y_val_full, y_val_store['group' + str(j)], axis=0)
    
        X_test_store = {}
        y_test_store = {}
        for j in range(1, k+1):
            test_data = full_data_transposed.iloc[range(num_train_samples + num_val_samples, len(full_data_transposed)), range(j-1, j)]
            X_test = np.zeros((num_test_samples-MIN, MIN, 1))
            y_test = np.zeros((num_test_samples-MIN, 1))
            for i in range(0, num_test_samples-MIN):
                X_test[i, -(MIN):, :] = test_data[i:i+MIN]
                y_test[i, :] = test_data[i+MIN:i+MIN+1]
            X_test_store["group" + str(j)] = X_test
            y_test_store["group" + str(j)] = y_test
        X_test_full = X_test_store['group' + str(1)]
        y_test_full = y_test_store['group' + str(1)]
        for j in range(2, k+1):
            X_test_full = np.append(X_test_full, X_test_store['group' + str(j)], axis=0)
            y_test_full = np.append(y_test_full, y_test_store['group' + str(j)], axis=0)
        return full_data_transposed, X_train_full, y_train_full, X_val_full, y_val_full, X_test_full, y_test_full
    
    def split_dataset_sequence_CNN_retrain(self, df_shuffle, MIN, k, ratio): # no zero padding
        MIN = MIN # the crucial difference with RNN, we only take the past MIN steps, this will also be the legth of vector
        k = k     # how many datasets we want to use
        ratio = ratio # the ratio of train-val-test split, the split is in "timesteps"
        full_data_transposed = df_shuffle.T

        num_train_samples = int(ratio[0] * len(full_data_transposed))
        num_val_samples = int(ratio[1] * len(full_data_transposed))
        num_test_samples = len(full_data_transposed) - num_val_samples - num_train_samples

        X_train_store = {}
        y_train_store = {}
        for j in range(1, k+1):
            train_data = full_data_transposed.iloc[range(0, num_train_samples), range(j-1, j)]
            X_train = np.zeros((num_train_samples-MIN, MIN, 1)) # length is MIN
            y_train = np.zeros((num_train_samples-MIN, 1))
            for i in range(0, num_train_samples-MIN):
                X_train[i, -(MIN):, :] = train_data[i:i+MIN] # the second argument means we take the last (i+MIN) values
                y_train[i, :] = train_data[i+MIN:i+MIN+1]        
            X_train_store["group" + str(j)] = X_train
            y_train_store["group" + str(j)] = y_train
        X_train_full = X_train_store['group' + str(1)]
        y_train_full = y_train_store['group' + str(1)]
        for j in range(2, k+1):
            X_train_full = np.append(X_train_full, X_train_store['group' + str(j)], axis=0)
            y_train_full = np.append(y_train_full, y_train_store['group' + str(j)], axis=0)

        X_test_store = {}
        y_test_store = {}
        for j in range(1, k+1):
            test_data = full_data_transposed.iloc[range(num_train_samples + num_val_samples, len(full_data_transposed)), range(j-1, j)]
            X_test = np.zeros((num_test_samples-MIN, MIN, 1))
            y_test = np.zeros((num_test_samples-MIN, 1))
            for i in range(0, num_test_samples-MIN):
                X_test[i, -(MIN):, :] = test_data[i:i+MIN]
                y_test[i, :] = test_data[i+MIN:i+MIN+1]
            X_test_store["group" + str(j)] = X_test
            y_test_store["group" + str(j)] = y_test
        X_test_full = X_test_store['group' + str(1)]
        y_test_full = y_test_store['group' + str(1)]
        for j in range(2, k+1):
            X_test_full = np.append(X_test_full, X_test_store['group' + str(j)], axis=0)
            y_test_full = np.append(y_test_full, y_test_store['group' + str(j)], axis=0)
        return full_data_transposed, X_train_full, y_train_full, X_test_full, y_test_full
    


def train_with_tuner_sequence_RNN_findepoch(path, hypermodel, save_path, model_name): 
    
    dataloader = DataLoader_Sequence(path)
    df_shuffle = dataloader.load_dataset_sequence(path)
    full_data_transposed, X_train, y_train, X_val, y_val, X_test, y_test = dataloader.split_dataset_sequence_CNN(df_shuffle, MIN, k, ratio_original)
    # note that the num_test_samples-MIN cannot be zero or below

    

    tuner = kt.BayesianOptimization(hypermodel, 
                                    objective="val_loss", 
                                    max_trials=max_trials, 
                                    executions_per_trial=executions_per_trial, 
                                    directory=save_path, 
                                    overwrite=True, 
                                    ) 

    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience),
                 keras.callbacks.ModelCheckpoint(model_name, save_best_only=True)] 

    tuner.search(x = X_train, y = y_train, 
                batch_size=batch_size, epochs=search_epochs, 
                validation_data=(X_val, y_val), 
                callbacks=callbacks, verbose=1,)
    tuner.results_summary()
    
    
    # Now we find the best epoch of the model by moitoring val_loss with EarlyStopping
    
    best_hps = tuner.get_best_hyperparameters(max_trials)  # we pick the best set of hyperparameters
    
    best_model_ep = hypermodel(best_hps[0])  # we pick the best set of hyperparameters
    
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=best_epoch_patience)] 
    

    best_model_ep.fit(x = X_train, y = y_train,
                epochs=best_epoch_training_epochs,
                validation_data=(X_val, y_val),
                batch_size=best_epoch_batch_size,
                callbacks=callbacks,
                verbose=1,
                )
    val_loss_per_epoch = best_model_ep.history.history["val_loss"]
    best_epoch = val_loss_per_epoch.index(min(val_loss_per_epoch)) + 1
    print(f"Best epoch: {best_epoch}")
    
    best_model_ep.summary()
    
    # We also plot the training and validation losses during the find best epoch process.
    plt.plot(best_model_ep.history.history["loss"], label="Training Loss")
    plt.plot(best_model_ep.history.history["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Losses")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()
    
    plt.plot(best_model_ep.history.history["mae"], label="Training MAE")
    plt.plot(best_model_ep.history.history["val_mae"], label="Validation MAE")
    plt.title("Training and Validation Metrics")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()

    return hypermodel, tuner, best_epoch, X_test, y_test, full_data_transposed



def sequence_load_tuner(hypermodel, load_dir, load_project):
    tuner = kt.BayesianOptimization(hypermodel,
                    objective="val_loss", 
                    directory=load_dir, 
                    project_name=load_project,
                    overwrite=False,
                    max_trials=max_trials, 
                    executions_per_trial=executions_per_trial, 
                    )
    return tuner


def sequence_load_best_hyperparam(hypermodel, load_dir, load_project, top_n):
    tuner = sequence_load_tuner(hypermodel, load_dir, load_project)
    return tuner.get_best_hyperparameters(top_n)

  # if you want to see the top_n hyper params, use below
  # best_trials = tuner.oracle.get_best_trials(num_trials=top_n)
  # for trial in best_trials:
    # trial.summary()
    # model = tuner.load_model(trial)
    # Do some stuff to the model



def sequence_train_with_tuner_ensemble_models(path, hypermodel, hp_dir, hp_proj): 
    # the saved checkpoint is under "hp_dir/hp_proj/"

    dataloader = DataLoader_Sequence(path)
    df_shuffle = dataloader.load_dataset_sequence(path)
    full_data_transposed, X_train, y_train, X_val, y_val, X_test, y_test = dataloader.split_dataset_sequence_CNN(df_shuffle, MIN, k, ratio_original)
    # note that the num_test_samples-MIN cannot be zero or below
    
    tuner = sequence_load_tuner(hypermodel, hp_dir, hp_proj)
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience)]
        
    #  keras.callbacks.ModelCheckpoint(model_name, save_best_only=True)] 
    tuner.search(x = X_train, y = y_train, 
                batch_size=batch_size, epochs=search_epochs, 
                validation_data=(X_val, y_val), 
                callbacks=callbacks, verbose=1,)
    tuner.results_summary()
    return tuner



#def sequence_model_retrain(path, hypermodel, tuner, best_epoch, retrain_patience, k, best_hps):
def sequence_model_retrain(path, hypermodel, retrain_patience, k, best_hps, save_model_path):
    # We take the best model found and retrain with the full training data.
    # We train multiple times and take the average of the best N models by looking at the overall relative errors.
    # Note that the Best_training_epochs will only be the maximum possible epoch, we will monitor by EarlyStopping with "loss".
    # We will only plot the training losses of the best_N models.
    
    dataloader = DataLoader_Sequence(path)
    df_shuffle = dataloader.load_dataset_sequence(path)
    full_data_transposed_old, X_train_old, y_train_old, X_val_old, y_val_old, X_test_old, y_test_old = dataloader.split_dataset_sequence_CNN(df_shuffle, MIN, k, ratio_original)
    full_data_transposed, X_train, y_train, X_test, y_test = dataloader.split_dataset_sequence_CNN_retrain(df_shuffle, MIN, k, ratio)
    # note that the num_test_samples-MIN cannot be zero or below
    
    final_test = pd.DataFrame(y_test)
    final_test.columns = ['Targets'] # The final targets
    
    ratio_retrain = len(X_train)/len(X_train_old) 
    
    #Best_training_epochs = int(best_epoch * ratio_retrain)
    
    model_retrain = {}
    model_history_retrain = {}
    df_store_test = {}
    
    #best_hps = tuner.get_best_hyperparameters(max_trials)
    
    callbacks = [keras.callbacks.EarlyStopping(monitor="loss", patience=retrain_patience, restore_best_weights=True)] 
    
    trained_models = {}
    j = 0
    while j < training_times:
        best_model_retrain = hypermodel(best_hps[0]) # We take again the best set of hyperparameters.
        best_model_retrain.fit(X_train, y_train, 
                      epochs=Best_training_epochs,
                      batch_size = batch_size,
                      callbacks=callbacks,
                      verbose=1,
                      )

        model_retrain[str(j)] = best_model_retrain
        model_history_retrain[str(j)] = best_model_retrain.history.history
        
        best_model_retrain.save(os.path.join(save_model_path, f"model_{j}.h5"))
        trained_models[j] = f"model_{j}.h5"


        test_predictions = best_model_retrain.predict(X_test) # for the original train-test split
        test_pred = pd.DataFrame(test_predictions)
        test_pred.columns = ['Model # '+ str(j) + ' Predictions']
        df_store_test[str(j)] = test_pred

        j = j + 1
        

    return trained_models, model_retrain, model_history_retrain, df_store_test, final_test
    
def sequence_model_average(trained_models, df_store_test, final_test, save_path):
    
    # We choose the best_N models by monitoring the overall relative errors 
    df_test_all = df_store_test[str(0)]
    for k in range(1, training_times):
        df_test_all = pd.concat([df_test_all, df_store_test[str(k)]], axis=1)

    df_compare = pd.concat([final_test, df_test_all], axis = 1)
    print(df_compare)
    for l in range(0, training_times):
        df_rel_error = (abs(df_compare['Targets']-df_compare['Model # '+ str(l) + ' Predictions'])/df_compare['Targets'])*100
        df_compare.insert(2+2*l, 'Rel Error (%) ' + str(l), df_rel_error)
    print(df_compare)
    
    column_sums = df_compare[['Rel Error (%) '+ str(m) for m in range(0, training_times)]].sum()
    smallest_columns = column_sums.nsmallest(best_N)
    
    print(column_sums)
    print(f"{smallest_columns.index} have the smallest relative errors.")
    
    column_indices = df_compare.columns.get_indexer(smallest_columns.index)

    # Select the columns one before each of these columns
    previous_columns = df_compare.iloc[:, column_indices - 1]
    
    # Find the model numbers
    column_names = list(previous_columns.columns)
    pattern = r'\d+'
    model_numbers = [int(re.search(pattern, column_name).group()) for column_name in column_names]
    
    mean_values = previous_columns.mean(axis=1)
    mean_values = pd.DataFrame(mean_values)
    mean_values.columns=['Model Average']
    df_compare_final = pd.concat([final_test, mean_values], axis=1)
    
    df_abs_error = abs(df_compare_final['Targets']-df_compare_final['Model Average'])
    df_rel_error = (abs(df_compare_final['Targets']-df_compare_final['Model Average'])/df_compare_final['Targets'])*100  
    
    df_compare_final.insert(2, 'Absolute Errors', df_abs_error)
    df_compare_final.insert(3, 'Relative Errors (%)', df_rel_error)
    
    print(df_compare_final) # The final predictions on the test data.
    
    plt.figure(figsize=(10,8))
    plt.plot(df_rel_error, '.')
    plt.title("Relative Errors (%) for Model Predictions")
    plt.show()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.distplot(df_compare_final['Relative Errors (%)'])
#     sns.kdeplot(df_compare_final['Relative Errors (%)'], ax=ax, shade=True, color='blue', alpha=0.05)


    ax.set_title('Density Plot of Relative Errors for Test Data')
    ax.set_xlabel('Relative Errors (%)')
    ax.set_ylabel('Density')
    fig.tight_layout()
    plt.savefig(os.path.join(save_path, 'TestDensity.jpg'))
    return df_compare, df_compare_final, model_numbers


def sequence_model_plots(model_history_retrain, model_numbers, save_path):
    # We only plot the losses of the best_N models, and will rename them in ascending order.
    
    x_range_list = []    
    legend_labels = []
    #fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=False, figsize=(20, 16))
    #fig, (ax1, ax3) = plt.subplots(2, 1, sharex=False, figsize=(20, 16))
    fig, ax1 = plt.subplots(1, 1, sharex=False, figsize=(20, 16))
    y_max = []
    y_max_mae = []
    
    i = 0
    while i < len(model_numbers):
        number = model_numbers[i]
        y_loss = model_history_retrain[str(number)]['loss']
        y_metrics = model_history_retrain[str(number)]["mae"]
        x = range(1,len(y_loss)+1)
        
        y_max.append(max(model_history_retrain[str(number)]["loss"][9:].copy()))
        y_max_mae.append(max(model_history_retrain[str(number)]["mae"][9:].copy()))

        ax1.plot(x, y_loss, '--')
        #ax2.plot(x, y_loss, '.--')

        #ax3.plot(x, y_metrics, '--')
        #ax4.plot(x, y_metrics, '.--')

        x_range_list.append(len(y_loss))
        
        legend_labels.append('Model #' + str(i) + ' (Epoch= '+ str(x_range_list[i])+ ')')

        i = i + 1

    fig.legend(legend_labels, loc="right", bbox_to_anchor=(1.16, 0.5), frameon=True, edgecolor='black', borderpad=1, labelspacing=2, handlelength=3) 
    ax1.set_title("Loss Function",fontsize=40)
    ax1.set_xlabel("Epoch", fontsize=25)
    ax1.set_ylabel("MSE", fontsize=25)
    ax1.set_xlim([1, max(x_range_list)])
    ax1.set_xticks(range(50,max(x_range_list),50))
    ax1.set_yscale('log')

#     ax2.set_title("Loss Function-truncated view")
#     ax2.set_xlabel("Epoch")
#     ax2.set_ylabel("MSE")
#     ax2.set_xlim([10, max(x_range_list)])
#     ax2.set_xticks(range(10,max(x_range_list),5))
#     ax2.set_ylim([0.0, 1.2*max(y_max)]) 
#     ax2.set_yscale('log')

#     ax3.set_title("Metric")
#     ax3.set_xlabel("Epoch")
#     ax3.set_ylabel("MAE")
#     ax3.set_xlim([1, max(x_range_list)])
#     ax3.set_xticks(range(5,max(x_range_list),5))
#     ax3.set_yscale('log')

#     ax4.set_title("Metrics-truncated view")
#     ax4.set_xlabel("Epoch")
#     ax4.set_ylabel("MAE")
#     ax4.set_xlim([10, max(x_range_list)])
#     ax4.set_xticks(range(10,max(x_range_list),5))
#     ax4.set_ylim([0.0, 1.2*max(y_max_mae)]) 
#     ax4.set_yscale('log')
    fig.tight_layout()
    plt.savefig(os.path.join(save_path, 'LossFunction.jpg'))
    
def prediction_unseen_data(trained_models, model_numbers, path, save_path): 
    # for predictions of the unseen data.
    # We start with the MIN steps of data, predict the MIN+1 step, but "not" reusing the prediction for next prediction. 
    dataloader = DataLoader_Sequence(path)
    df_shuffle = dataloader.load_dataset_sequence(path)
    full_data_transposed, X_train, y_train, X_val, y_val, X_test, y_test = dataloader.split_dataset_sequence_CNN(df_shuffle, MIN, k, ratio_original)

    num_train_samples = int(len(full_data_transposed))

    X_train_store = {}
    y_train_store = {}
    for j in range(k+1, len(full_data_transposed.columns)+1):
        train_data = full_data_transposed.iloc[range(0, num_train_samples), range(j-1, j)]
        X_train = np.zeros((num_train_samples-MIN, MIN, 1)) # length is MIN
        y_train = np.zeros((num_train_samples-MIN, 1))
        for i in range(0, num_train_samples-MIN):
            X_train[i, -(MIN):, :] = train_data[i:i+MIN] # the second argument means we take the last (i+MIN) values
            y_train[i, :] = train_data[i+MIN:i+MIN+1]        
        X_train_store["group" + str(j)] = X_train
        y_train_store["group" + str(j)] = y_train
    X_train_full = X_train_store['group' + str(k+1)]
    y_train_full = y_train_store['group' + str(k+1)]
    for j in range(k+2, len(full_data_transposed.columns)+1):
        X_train_full = np.append(X_train_full, X_train_store['group' + str(j)], axis=0)
        y_train_full = np.append(y_train_full, y_train_store['group' + str(j)], axis=0)

    predictions_set = {}
    j = 0
    while j < len(model_numbers):
        number = model_numbers[j]
        model = tf.keras.models.load_model(trained_models[number])

        predictions = model.predict(X_train_full)
        pred = pd.DataFrame(predictions)
        
        predictions_set[str(j)] = pred
        j = j + 1
    
    df_predictions = predictions_set[str(0)]
    for l in range(1, len(model_numbers)):
        pred = pd.concat([df_predictions, predictions_set[str(l)]], axis=1)
        
    pred = pred.mean(axis=1) 
    pred = pd.DataFrame(pred)
    pred.columns = ['Model Predictions']
    test = pd.DataFrame(y_train_full)
    test.columns = ['Targets']
    df_compare = pd.concat([test, pred], axis = 1)
    print(df_compare)
    
    df_abs_error = abs(df_compare['Targets']-df_compare['Model Predictions'])
    df_rel_error = (abs(df_compare['Targets']-df_compare['Model Predictions'])/df_compare['Targets'])*100  
    df_compare1 = pd.concat([df_abs_error, df_rel_error], axis = 1)
    df_compare1.columns = ['Abs Error for Model', 'Rel Error for Model']
    print(df_compare1)
    
    plt.figure(figsize=(10,8))
    plt.plot(df_rel_error, '.')
    plt.title("Relative Errors for Model Predictions")
    plt.show()
    
    fig, ax = plt.subplots(figsize=(10, 8))
#     sns.kdeplot(df_compare_final['Relative Errors (%)'], ax=ax, shade=True, color='blue', alpha=0.05)
    sns.distplot(df_compare1['Rel Error for Model'])


    ax.set_title('Density Plot of Relative Errors for Unseen Data')
    ax.set_xlabel('Relative Errors (%)')
    ax.set_ylabel('Density')
    fig.tight_layout()
    plt.savefig(os.path.join(save_path, 'UnseenDensity.jpg'))
    
    return df_compare, df_compare1


def prediction_sequence(trained_models, model_numbers, path, n): # predict the next n values 
    # This will always be using the past MIN data to predict the next value.
    dataloader = DataLoader_Sequence(path)
    df_shuffle = dataloader.load_dataset_sequence(path)
    full_data_transposed, X_train, y_train, X_val, y_val, X_test, y_test = dataloader.split_dataset_sequence_CNN(df_shuffle, MIN, k, ratio_original)
    
    sequence_test = np.zeros((k, MIN, 1))
    for m in range(1, k+1):
        sequence_update = full_data_transposed.iloc[-(MIN):, range(m-1, m)]
        sequence_test[m-1, -(MIN):, :] = sequence_update[0:MIN]
    

    
    predictions_set = {}
    j = 0
    while j < len(model_numbers):
        number = model_numbers[j]
        model = tf.keras.models.load_model(trained_models[number])

        predictions = model.predict(sequence_test)
        pred = pd.DataFrame(predictions)
        
        predictions_set[str(j)] = pred
        j = j + 1
    
    df_predictions = predictions_set[str(0)]
    for l in range(1, len(model_numbers)):
        pred = pd.concat([df_predictions, predictions_set[str(l)]], axis=1)
        
    pred = pred.mean(axis=1) 
    pred = pd.DataFrame(pred)
    pred.columns = [len(full_data_transposed)+1]
    pred = pred.T
    sequence_update = pd.concat([full_data_transposed.iloc[range(0, len(full_data_transposed)), range(0, k)], pred], axis = 0)
    
    i = 2
    while i <= n:
        sequence_test1 = np.zeros((k, MIN, 1))
        for j in range(1, k+1):
            sequence_update1 = sequence_update.iloc[-(MIN):, range(j-1, j)]
            sequence_test1[j-1, -(MIN):, :] = sequence_update1[0:MIN]
        
        predictions_set = {}
        j = 0
        while j < len(model_numbers):
            number = model_numbers[j]
            model = tf.keras.models.load_model(trained_models[number])

            predictions = model.predict(sequence_test1)
            pred = pd.DataFrame(predictions)

            predictions_set[str(j)] = pred
            j = j + 1

        df_predictions = predictions_set[str(0)]
        for l in range(1, len(model_numbers)):
            pred = pd.concat([df_predictions, predictions_set[str(l)]], axis=1)

        pred = pred.mean(axis=1) 
        pred = pd.DataFrame(pred)
        pred.columns = [len(full_data_transposed)+i]
        pred = pred.T
        sequence_update = pd.concat([sequence_update, pred], axis = 0)
            
        i = i + 1
    return sequence_update


# In[ ]:


MIN = 5 # the length of each vector
k = 1000 # how many datasets we want to use out of the full 10000 sets
ratio_original = (0.6, 0.2, 0.2)

# load the datasets
dataloader = DataLoader_Sequence('')
df_shuffle = dataloader.load_dataset_sequence('')
full_data_transposed, X_train, y_train, X_val, y_val, X_test, y_test = dataloader.split_dataset_sequence_CNN(df_shuffle, MIN, k, ratio_original)


# In[ ]:


train_data_path = ''
save_hp_dir = ''
save_hp_proj = ''
save_model_path = ''
save_fig_dir = ''

# this is to assure the path name is correct
import os
assert(os.path.exists(train_data_path))
assert(os.path.exists(test_data_path))
assert(os.path.exists(save_hp_dir))
assert(os.path.exists(save_model_path))
assert(os.path.exists(save_fig_dir))


# In[ ]:


# One could run the following line by line for each example considered in section 4 of arXiv:2305.00997

tuner = sequence_train_with_tuner_ensemble_models(train_data_path, hyper_RNN_model, save_hp_dir, save_hp_proj)

best_hps = sequence_load_best_hyperparam(hyper_RNN_model, save_hp_dir, save_hp_proj, max_trials)
trained_models, model_retrain, model_history_retrain, df_store_test, final_test = sequence_model_retrain(train_data_path, hyper_RNN_model, retrain_patience, k, best_hps, save_model_path)

best_N = 2
df_compare, df_compare_final, model_numbers = sequence_model_average(trained_models, df_store_test, final_test, save_fig_dir)

sequence_model_plots(model_history_retrain, model_numbers, save_fig_dir)

df_compare, df_compare1 = prediction_unseen_data(trained_models, model_numbers, train_data_path, save_fig_dir)


# In[ ]:


# generating figures for the relative errors of test data

fig_dist, ax3 = plt.subplots(1,1,figsize=(10,5))
sns.distplot(df_compare_final['Relative Errors (%)'])
ax3.set_title('Density Plot of Relative Errors for the p Test Data')
ax3.set_xlabel('Relative Errors (%)')
ax3.set_xlim(left=-1, right=10)
ax3.set_ylabel('Density')
plt.savefig(save_fig_dir+"/relative_error_density1.jpg")

fig_dist, ax3 = plt.subplots(1,1,figsize=(10,5))
# sns.distplot(df_compare_final['Relative Errors (%)'])
sns.distplot(df_compare1['Rel Error for Model'], color="green")
ax3.set_title('Density Plot of Relative Errors for the q Test Data')
ax3.set_xlabel('Relative Errors (%)')
ax3.set_xlim(left=-2, right=25)
ax3.set_ylabel('Density')
plt.savefig(save_fig_dir+"/relative_error_density3.jpg")


# ### The following provides the source code for reproducing results in section 5 of arXiv: 2305.00997 of studying the expressivity of von Neumann entropy using the Fourier series representation of the generating function. This is based on PennyLane. 

# In[ ]:


import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as np

# plt.style.reload_library()
# plt.style.use(['science','no-latex'])

plt.rcParams['figure.dpi'] = 500

np.random.seed(42)

def square_loss(targets, predictions):
    loss = 0
    for t, p in zip(targets, predictions):
        loss += (t - p) ** 2
    loss = loss / len(targets)
    return 0.5*loss

def target_Fourier_series(x): # generate the target Fourier series with input above
    series = coeff0 # initialize the series
    for order, coeff in order_coeffs:
        exponent = np.complex128(scaling * order * x * 1j)
        conj_coeff = np.conjugate(coeff)
        series += coeff * np.exp(exponent) + conj_coeff * np.exp(-exponent)
    return np.real(series)

def target_Fourier_series_nonReal(x): 
  # generate the target Fourier series for the case of non-real function
  # note also that we need to use 2pi for each order corresponding to the Fourier parameter
    series = coeff0 # initialize the series
    for order, coeff in order_coeffs:
        exponent = np.complex128(scaling * 2 * np.pi * order * x * 1j)
        series += coeff * np.exp(exponent)
    return np.real(series)

def target_Fourier_series_stretched(x): # For stretched interval 
    series = coeff0 # initialize the series
    L = 1
    for order, coeff in order_coeffs:
        exponent = np.complex128(scaling * 2 * np.pi * order * x * 1j/L)
        conj_coeff = np.conjugate(coeff)
        series += coeff * np.exp(exponent)/np.sqrt(L) + conj_coeff * np.exp(-exponent)/np.sqrt(L)
    return np.real(series)


# In[ ]:


Using the generating function as the Fourier series

# degree = 1  # degree of the target function 
scaling = 1  # scaling of the data
#order_coeffs = [(1, -0.196497), (2, -0.0752353), (3, -0.0592326), (4, -0.0395808)] # coefficients of non-zero frequencies, c1, c2,..etc.

# We choose the parameters for the coefficients to be L=2, epsilon=0.1

order_coeffs = [(1, 0.028467 + 0.131412j), (2, 0.010255 + 0.070899j), (3, 0.005445 + 0.048685j), (4, 0.003413 + 0.037097j)] # coefficients of non-zero frequencies, c1, c2,..etc. # For the generating function
coeff0 = 0.35978  # coefficient of zero frequency

# Note that we need to change the interval in accordance with the stretched case
x = np.linspace(0, 1, 300, requires_grad=False) # requires_grad=False where the parameters are not considered as trainable

# x1 = np.linspace(0,0.2,50,  requires_grad=False)
# x2 = np.linspace(0.2, 0.8, 100, requires_grad=False)
# x3 = np.linspace(0.8, 1.0, 50, requires_grad=False)
# x = np.concatenate((x1, x2, x3))
target_y = np.array([target_Fourier_series_stretched(x_) for x_ in x], requires_grad=False)

#plt.plot(x, target_y, c='black') #, c='red'
plt.scatter(x, target_y, linewidths=1 , facecolor='white', edgecolor='red', marker='.', alpha=1.0) #, facecolor='white', edgecolor='black'
plt.ylim(np.amin(target_y)-0.1, np.amax(target_y)+0.1)
plt.show();


# In[ ]:


# Serial Model

dev = qml.device('default.qubit', wires=1)

def S(x):
    """Data-encoding circuit block."""
    qml.RX(scaling * x, wires=0)

def W(theta):
    """Trainable circuit block."""
    qml.Rot(theta[0], theta[1], theta[2], wires=0)


@qml.qnode(dev)
def serial_quantum_model(weights, x):

    for theta in weights[:-1]:
        W(theta)
        S(x)

    # (L+1)'th unitary
    W(weights[-1])

    return qml.expval(qml.PauliZ(wires=0))
    #PauliZ meansure will always restrict the y-range to be [-1,1] since the eigenvalues are [-1,1]


# In[ ]:


# We can run the following multiple times, each time sampling different weights, and therefore different quantum models.

r = 6 # number of times the encoding gets repeated (here equal to the number of layers) 
# this value should be the same as the order of Fourier series
weights = 2 * np.pi * np.random.random(size=(r+1, 3), requires_grad=True) # some random initial weights
scaling_model = 6

# Note that we need to change the interval and the number of samples below in accordance with the target
x = np.linspace(0, scaling_model, 300, requires_grad=False)

# x1 = np.linspace(0, 0.2,50,  requires_grad=False)
# x2 = np.linspace(0.2, 0.8, 100, requires_grad=False)
# x3 = np.linspace(0.8, 1.0, 50, requires_grad=False)
# x_con = np.concatenate((x1, x2, x3))
# x = scaling_model * ((x_con - x_con.min()) / (x_con.max() - x_con.min()))
random_quantum_model_y = [serial_quantum_model(weights, x_) for x_ in x]



fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(x, random_quantum_model_y, c='black')
ax.set_ylim(np.amin(random_quantum_model_y)-0.1, np.amax(random_quantum_model_y)+0.1)
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
plt.show()

print(qml.draw(serial_quantum_model)(weights, x[-1]))


# In[ ]:


def cost(weights, x, y):
    predictions = [serial_quantum_model(weights, x_) for x_ in x]
    return square_loss(y, predictions)

max_steps = 1500
opt = qml.AdamOptimizer(0.005)
batch_size = 100
cst = [cost(weights, x, target_y)]  # initial cost
best_weights = weights
best_cost = cst[0]
best_step = 0

for step in range(max_steps):

    # Select batch of data
    batch_index = np.random.randint(0, len(x), (batch_size,))
    x_batch = x[batch_index]
    y_batch = target_y[batch_index]

    # Update the weights by one optimizer step
    weights, _, _ = opt.step(cost, weights, x_batch, y_batch)

    # Save, and possibly print, the current cost
    c = cost(weights, x, target_y)
    cst.append(c)
    if (step + 1) % 1 == 0:
        print("Cost at step {0:3}: {1}".format(step + 1, c))
    if c < best_cost:
        best_weights = weights
        best_cost = c
        best_step = step + 1

#print("Best weights: ", best_weights)
print("Best cost: ", best_cost)
print("Found at step: ", best_step)


# In[ ]:


fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(range(len(cst)), cst, '--', color="red")
ax.set_xlim(0,983)
# ax.set_ylim(0, np.amax(cst)*1.1)
ax.set_title('Loss Function')
ax.set_xlabel('Epochs')
xticks = list(range(200,best_step,200))
xticks.append(best_step)
ax.set_xticks(xticks)
ax.set_ylabel('MSE')
ax.set_yscale('log')
plt.show()


# In[ ]:


predictions = [serial_quantum_model(best_weights, x_) for x_ in x]


fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(x, target_y, c='black')
ax.scatter(x, target_y, facecolor='white', edgecolor='black')
ax.plot(x, predictions, c='red')
ax.set_ylim(np.amin(target_y)-0.1, np.amax(target_y)+0.1)
#ax.set_title('Density Plot of Relative Errors for Test Data')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
fig.tight_layout()
plt.show();


# In[ ]:


# Parallel Model

from pennylane.templates import StronglyEntanglingLayers

# scaling = 1
r = 5

dev = qml.device('default.qubit', wires=r)

def S(x):
    """Data-encoding circuit block."""
    for w in range(r):
        qml.RX(scaling * x, wires=w)

def W(theta):
    """Trainable circuit block."""
    StronglyEntanglingLayers(theta, wires=range(r))


@qml.qnode(dev)
def parallel_quantum_model(weights, x):

    W(weights[0])
    S(x)
    W(weights[1])

    return qml.expval(qml.PauliZ(wires=0))


# In[ ]:


trainable_block_layers = 3
weights = 2 * np.pi * np.random.random(size=(2, trainable_block_layers, r, 3), requires_grad=True)
model_scaling = 6

x = np.linspace(0, model_scaling, 300, requires_grad=False)
random_quantum_model_y = [parallel_quantum_model(weights, x_) for x_ in x]

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(x, random_quantum_model_y, c='black')
ax.set_ylim(np.amin(random_quantum_model_y)-0.1, np.amax(random_quantum_model_y)+0.1)
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
plt.show()

print(qml.draw(parallel_quantum_model)(weights, x[-1]))


# In[ ]:


def cost(weights, x, y):
    predictions = [parallel_quantum_model(weights, x_) for x_ in x]
    return square_loss(y, predictions)

max_steps = 1000
opt = qml.AdamOptimizer(0.005)
batch_size = 100
cst = [cost(weights, x, target_y)]  # initial cost
best_weights = weights
best_cost = cst[0]
best_step = 0

for step in range(max_steps):

    # Select batch of data
    batch_index = np.random.randint(0, len(x), (batch_size,))
    x_batch = x[batch_index]
    y_batch = target_y[batch_index]

    # Update the weights by one optimizer step
    weights, _, _ = opt.step(cost, weights, x_batch, y_batch)

    # Save, and possibly print, the current cost
    c = cost(weights, x, target_y)
    cst.append(c)
    if (step + 1) % 1 == 0:
        print("Cost at step {0:3}: {1}".format(step + 1, c))
    if c < best_cost:
        best_weights = weights
        best_cost = c
        best_step = step + 1

#print("Best weights: ", best_weights)
print("Best cost: ", best_cost)
print("Found at step: ", best_step)


# In[ ]:


fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(range(len(cst)), cst, '--', color="red")
ax.set_xlim(0,917)
# ax.set_ylim(0, np.amax(cst)*1.1)
ax.set_title('Loss Function')
ax.set_xlabel('Epochs')
xticks = list(range(200,best_step,200))
xticks.append(best_step)
ax.set_xticks(xticks)
ax.set_ylabel('MSE')
ax.set_yscale('log')
plt.show()


# In[ ]:


predictions = [parallel_quantum_model(best_weights, x_) for x_ in x]


fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(x, target_y, c='black')
ax.scatter(x, target_y, facecolor='white', edgecolor='black')
ax.plot(x, predictions, c='red')
ax.set_ylim(np.amin(target_y)-0.1, np.amax(target_y)+0.1)
#ax.set_title('Density Plot of Relative Errors for Test Data')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
fig.tight_layout()
plt.show();

