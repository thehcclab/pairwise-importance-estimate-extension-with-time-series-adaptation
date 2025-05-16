import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from typing import List
from utilities.Dataset import *

def user_fold_load(idx: int, user_fold: dict, participants: List, batch_size=1, selection=None, transpose_channels=False, debug=False):

    y_fold= participants[idx]
    x_fold= list(participants[:idx])+list(participants[idx+1:])

    X_val, y_val= user_fold[y_fold]
    if transpose_channels:
        X_val = X_val.swapaxes(1,2)    

    if debug:
        print(f"X_val  : {X_val.shape}, y_val  : {y_val.shape}")

    X_train=[]
    y_train=[]
    for fold in x_fold:
        X_train.append(user_fold[fold][0])
        y_train.append(user_fold[fold][1])

    X_train= np.concatenate(X_train)
    if transpose_channels:
        X_train = X_train.swapaxes(1,2)      
    y_train= np.concatenate(y_train)
    if debug:
        print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")

    if selection is None:
        dataset_train= SubsetDataset(X_train, y_train)
        dataset_val= SubsetDataset(X_val, y_val)
    else:
        dataset_train= SubsetDataset(X_train[:, :,selection], y_train)
        dataset_val= SubsetDataset(X_val[:, :,selection], y_val)

    train_dataloader= torch.utils.data.DataLoader(dataset_train, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4)
    val_dataloader= torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, pin_memory=True, num_workers=4)

    classes= np.unique(list(y_train)+list(y_val))
#     print(classes)
    
    _, counts= np.unique(y_train, return_counts=True)
    class_ratio= counts/ counts.sum()
#     classes, counts= np.unique(list(np.unique(y_train))+list(np.unique(y_val)), return_counts=True)
#     print(counts)
#     print(counts/ counts.sum())

    return train_dataloader, val_dataloader, len(classes), X_train.shape[1:], class_ratio

def userfold_results_summary(fold_dictionaries,participants):
    dictionaries= []

    for i in range(len(participants)):
    #     dictionaries.append({'participant':participants[i], 'accuracy':fold_average[i]['accuracy'], 'f1-score':fold_average[i]['weighted avg']['f1-score']})
        dictionaries.append({'accuracy':fold_dictionaries[i]['accuracy'], 'f1-score':fold_dictionaries[i]['weighted avg']['f1-score']})

    sns.heatmap(pd.DataFrame(dictionaries, index=participants), vmin=0, vmax=1)
    
    print(pd.DataFrame(dictionaries, index=participants))
    plt.show()

def userfold_classwise_results_summary(fold_dictionaries, participants):
    dictionaries= []

    for i in range(len(participants)):
    #     dictionaries.append({'participant':participants[i], 'accuracy':fold_average[i]['accuracy'], 'f1-score':fold_average[i]['weighted avg']['f1-score']})
        dictionaries.append({'class0 f1-score':fold_dictionaries[i]['0']['f1-score'], 'class1 f1-score':fold_dictionaries[i]['1']['f1-score']})

    sns.heatmap(pd.DataFrame(dictionaries, index=participants), vmax=1, vmin=0)
    
    print(pd.DataFrame(dictionaries, index=participants))
    plt.show()