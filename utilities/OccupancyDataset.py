import numpy as np
from utilities.Dataset import *
from utilities.preproc_func import *

import pandas as pd
import math
from io import StringIO
import pickle

def split_seq(X, seq_len=64): # stride split
    array=[]

    for i in range(0,len(X)-seq_len):
        array.append(X[i:i+seq_len])

    return np.array(array)

def create_seq_indices(Y, split=0.4):
    """splitting each class approximately equally"""

    indices= [i for i in range(len(Y))]
    np.random.shuffle(indices)

    split_indices= int(math.floor(len(Y) * split))

    train_indices= np.array(indices[split_indices:])
    val_indices= np.array(indices[:split_indices])

    return train_indices, val_indices

from sklearn.preprocessing import StandardScaler, MinMaxScaler
class OccupancyDataset(MultivariateDataset):

    def __init__(self,config:dict,default_path="./experiments/occupancy/"): #init or load
    # assert protocol=="init" or protocol=="load", "\"load\" or \"init\""
        super().__init__()

        split= config["split"]

        with open("Occupancy/datatraining.txt") as file:
            data= file.read()

        df= pd.read_csv(StringIO(data),sep=",",
                        header=0, index_col=0,engine="python")
        df= df.drop("date",axis=1)

        scaler= MinMaxScaler()
        for i in df.keys()[:-1]:
            df[i]= scaler.fit_transform(df[i].to_numpy().reshape(-1,1))

        # split= len(df) * split
        self.classes= len(df["Occupancy"].unique())
        self.X= df.drop("Occupancy", axis=1).to_numpy()
        self.y= df["Occupancy"].to_numpy().astype(int)

        # print("before",self.X.shape, self.y.shape)
        self.X= split_seq(self.X, config["seq_len"])
        self.y= split_seq(self.y, config["seq_len"])
        # print("after",self.X.shape, self.y.shape)

        try:
            self.train_indices, self.val_indices= pickle.load(open(f"{default_path}indices.pkl", "rb"))
        except FileNotFoundError:
            self.train_indices, self.val_indices= create_seq_indices(self.y, split=split)
            pickle.dump([self.train_indices, self.val_indices], open(f"{default_path}indices.pkl", "wb"))

        # print(len(self.train_indices), len(self.val_indices))
        self.X_train= self.X[self.train_indices]; self.X_val= self.X[self.val_indices]
        self.y_train= self.y[self.train_indices]; self.y_val= self.y[self.val_indices]
