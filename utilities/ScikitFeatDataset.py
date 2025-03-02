import pandas as pd
import numpy as np
import math
import torch
from scipy.io import loadmat
import pickle

from utilities.preproc_func import *
from utilities.Dataset import *
# class ScikitFeatSubsetDataset(torch.utils.data.Dataset):
#
#     def __init__(self, X:np.array, y:np.array):
#
#         self.X= X.astype(float)
#         self.y= y.astype(int)
#
#     def __len__(self):
#         return len(self.X)
#
#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]

class ScikitFeatDataset(Dataset):

    def __init__(self, dataset:str, split=0.4, default_path="./experiments/scikit_feature/"):

        self.default_path= default_path+dataset

        self.dataset=dataset

        mat = loadmat('./scikit_feature/skfeature/data/'+dataset+'.mat')

        self.X= mat["X"]
        Y, self.classes= convert_labels(mat["Y"])
        self.y= Y.reshape(-1)
        self.classes= len(np.unique(self.y))
        self.split= split

        try:
            self.train_indices, self.val_indices= pickle.load(open(self.default_path+"-indices.pkl", "rb"))
        except FileNotFoundError:
            self.save()

        self.X_train= self.X[self.train_indices]; self.X_val= self.X[self.val_indices]
        self.y_train= self.y[self.train_indices]; self.y_val= self.y[self.val_indices]
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]

    def save(self):
        self.train_indices, self.val_indices= create_indices(self.y, self.split)

        pickle.dump([self.train_indices, self.val_indices], open(self.default_path+"-indices.pkl", "wb"))

    def return_indices(self):
        return self.train_indices, self.val_indices
