import torch
import numpy as np
from utilities.Dataset import SubsetDataset
import matplotlib.pyplot as plt
from typing import List
import os
import pickle

def stratified_10_fold_plot(path:str, dictionaries:List, save_path:str, folds= 10, save=True):
    # folds=10;
    labels=6; gt=["bike", "sit", "stand", "walk", "stairsup", "stairsdown"];
    fig, ax= plt.subplots(1,10)
    fig.set_figwidth(25)
    fig.set_figheight(6)

    for j in range(labels):
            ax[0].bar(j+1, dictionaries[0][str(j)]['f1-score'])
            ax[0].set_xticks([i+1 for i in range(labels)])
            ax[0].set_xticklabels(gt, rotation=270)


    for i in range(1,folds):
        for j in range(labels):
            ax[i].bar(j+1, dictionaries[i][str(j)]['f1-score'])
        ax[i].set_yticks([])
        ax[i].set_xticks([k+1 for k in range(labels)])
        ax[i].set_xticklabels(gt, rotation=270)


    fig.suptitle("stratified_10_fold-F1 Score", y=0.93)
    if save: #{feat}-{config['epoch']}
        if not os.path.exists(f"{path}/graphs/"):
            os.makedirs(f"{path}/graphs/")
        fig.savefig(path+f"/graphs/{save_path}.png")
    plt.close(fig)


def stratified_10_fold_load(i: int, stratified_10_fold: dict, batch_size=1, selection=None):

    y_fold= i#np.random.randint(10)
    tmp= np.arange(10)
    x_fold= list(tmp[:y_fold])+list(tmp[y_fold+1:])

    X_train=[]
    y_train=[]
    X_val=[]
    y_val=[]

    for instances in stratified_10_fold[y_fold]:
        X_val.append(instances[0])
        y_val.append([instances[1] for i in range(len(instances[0])) ])
        # y_val.append(instances[1])
    X_val= np.array(X_val)
    y_val= np.array(y_val)

    for fold in x_fold:
        for instances in stratified_10_fold[fold]:
            X_train.append(instances[0])
            y_train.append([instances[1] for i in range(len(instances[0])) ])
            # y_train.append(instances[1])
    X_train= np.array(X_train)
    y_train= np.array(y_train)

    if selection==None:
        dataset_train= SubsetDataset(X_train, y_train)
        dataset_val= SubsetDataset(X_val, y_val)
    else:
        dataset_train= SubsetDataset(X_train[:, selection], y_train)
        dataset_val= SubsetDataset(X_val[:, selection], y_val)

    train_dataloader= torch.utils.data.DataLoader(dataset_train, shuffle=True, batch_size=batch_size)
    val_dataloader= torch.utils.data.DataLoader(dataset_val, batch_size=batch_size)

    classes= len(np.unique(list(np.unique(y_train))+list(np.unique(y_val))))

    return train_dataloader, val_dataloader, classes, X_train.shape[2]

def user_fold_load(idx: int, user_fold: dict, USERS: List, batch_size=1, selection=None):

    y_fold= USERS[idx]
    x_fold= list(USERS[:idx])+list(USERS[idx+1:])

    X_train=[]
    y_train=[]
    X_val=[]
    y_val=[]

    for instances in user_fold[y_fold]:
        X_val.append(instances[0])
        y_val.append([instances[1] for i in range(len(instances[0])) ])
        # y_val.append(instances[1])
    X_val= np.array(X_val)
    y_val= np.array(y_val)

    for fold in x_fold:
        for instances in user_fold[fold]:
            X_train.append(instances[0])
            y_train.append([instances[1] for i in range(len(instances[0])) ])
            # y_train.append(instances[1])
    X_train= np.array(X_train)
    y_train= np.array(y_train)

    if selection is None:
        dataset_train= SubsetDataset(X_train, y_train)
        dataset_val= SubsetDataset(X_val, y_val)
    else:
        dataset_train= SubsetDataset(X_train[:, :,selection], y_train)
        dataset_val= SubsetDataset(X_val[:, :,selection], y_val)

    train_dataloader= torch.utils.data.DataLoader(dataset_train, shuffle=True, batch_size=batch_size)
    val_dataloader= torch.utils.data.DataLoader(dataset_val, batch_size=batch_size)

    classes= len(np.unique(list(np.unique(y_train))+list(np.unique(y_val))))

    return train_dataloader, val_dataloader, classes, X_train.shape[2]

def user_fold_plot(path:str, dictionaries:List, save_path:str, users=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'], save=True):

    # users=;
    labels=6;x=2; y=5
    gt=["bike", "sit", "stand", "walk", "stairsup", "stairsdown"];
    fig, ax= plt.subplots(2,5)
    fig.set_figwidth(25)
    fig.set_figheight(10)
    count=0

    for i in range(x):
        for j in range(y):
            for k in range(labels):
                ax[i][j].bar(k+1, dictionaries[count][str(k)]['f1-score'])
    #         ax[i][j].set_yticks([])
            ax[i][j].set_xticks([k+1 for k in range(labels)])
            ax[i][j].set_xticklabels(gt, rotation=270)
            ax[i][j].set_title(f"User {users[count]}")
            count += 1
            if count==len(users):
                break

    for i in range(x):
        for j in range(1,y):
            ax[i][j].set_yticks([])

    fig.delaxes(ax[1][-1])
    fig.suptitle("user_fold-F1 Score", y=0.93)
    if save: #{feat}-{config['epoch']}
        if not os.path.exists(f"{path}/graphs/"):
            os.makedirs(f"{path}/graphs/")
        fig.savefig(path+f"/graphs/{save_path}.png")
    plt.close(fig)

def HHAR_post(dict_path, state_path, fold, dictionary, model, feat, epoch):
    if not os.path.exists(dict_path):
        os.makedirs(dict_path)

    pickle.dump(dictionary, open(f"{dict_path}/{feat}-dictionary-fold{fold}-e{epoch}.pkl","wb"))

    if not os.path.exists(state_path):
        os.makedirs(state_path)

    torch.save(model.state_dict(), f"{state_path}/{feat}-state_dict-fold{fold}-e{epoch}.pt")
