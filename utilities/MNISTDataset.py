import pickle
import numpy as np
import torch
from mnist import MNIST

from utilities.Dataset import *

class MNISTDataset(torch.utils.data.Dataset):

    def __init__(self, subset:str, path="./MNIST/raw"):

        assert subset=="TRAIN" or subset=="TEST", "subset should either be TRAIN or TEST"

        mndata= MNIST(path, return_type="numpy")

        if subset=="TRAIN":
            self.X, self.y= mndata.load_training()
        elif subset=="TEST":
            self.X, self.y= mndata.load_testing()

        self.classes= len(np.unique(self.y))

    def return_subset_dataset(self,selections):
        return SubsetDataset(self.X[:, selections], self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
