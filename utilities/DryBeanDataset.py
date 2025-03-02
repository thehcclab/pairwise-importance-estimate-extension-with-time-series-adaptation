import torch
from sklearn.preprocessing import MinMaxScaler
import pickle
import numpy as np
import pandas as pd
from utilities.preproc_func import *
from io import StringIO

def convert_categorical_labels(df:pd.DataFrame, column="y"):
    count=0
    for i in np.unique(df[column]):
        df.loc[df[column]==i, column]=count
        count += 1
    return df

from utilities.Dataset import *
# class SubsetDataset(torch.utils.data.Dataset):
#     def __init__(self, X, y):
#         super().__init__()
#
#         self.X= X.astype(float)
#         self.y= y.astype(int)
#
#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]
#
#     def __len__(self):
#         return len(self.X)

class DryBeanDataset(Dataset):

    def __init__(self, dataset_path="DryBeanDataset/Dry_Bean_Dataset.arff", split=0.4, default_path="./experiments/DryBean/"):
        with open(dataset_path) as file:
            data= file.read()
        df= pd.read_csv(StringIO(data), skiprows=25,sep=",",
                        names=["Area","Perimeter","MajorAxisLength","MinorAxisLength","AspectRation",
                               "Eccentricity","ConvexArea","EquivDiameter","Extent","Solidity","roundness","Compactness",
                               "ShapeFactor1", "ShapeFactor2","ShapeFactor3","Shapefactor4","y"], engine="python")

        scaler = MinMaxScaler()

        df= convert_categorical_labels(df)

        self.X = scaler.fit_transform(df.drop("y", axis=1).to_numpy())
        self.y = df["y"].to_numpy()
        self.classes= len(np.unique(df["y"]))

        try:
            self.train_indices, self.val_indices= pickle.load(open(f"{default_path}indices.pkl", "rb"))
        except Exception:
            self.train_indices, self.val_indices= create_indices(self.y, split)
            pickle.dump([self.train_indices, self.val_indices], open(f"{default_path}indices.pkl","wb"))

        self.X_train= self.X[self.train_indices]; self.X_val= self.X[self.val_indices]
        self.y_train= self.y[self.train_indices]; self.y_val= self.y[self.val_indices]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
