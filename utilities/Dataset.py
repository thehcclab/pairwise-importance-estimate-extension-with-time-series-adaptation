import torch
import pandas as pd

class SubsetDataset(torch.utils.data.Dataset):
    def __init__(self, X:pd.DataFrame, y:pd.DataFrame):

        super().__init__()

        self.X= X
        self.y= y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()

        self.X_train= None; self.y_train= None
        self.X_val= None; self.y_val= None

    def return_training_dataset(self):
        return SubsetDataset(self.X_train, self.y_train)

    def return_training_subset_dataset(self,selections):
        return SubsetDataset(self.X_train[:, selections], self.y_train)

    def return_validation_dataset(self):
        return SubsetDataset(self.X_val, self.y_val)

    def return_validation_subset_dataset(self,selections):
        return SubsetDataset(self.X_val[:, selections], self.y_val)

    def return_y_val(self):
        return self.y_val

    def return_training_data(self):
        return [self.X_train, self.y_train]

class MultivariateDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()

        self.X_train= None; self.y_train= None
        self.X_val= None; self.y_val= None

    def return_training_dataset(self):
        return SubsetDataset(self.X_train, self.y_train)

    def return_training_subset_dataset(self,selections,axis=2):
        if axis==2:
            return SubsetDataset(self.X_train[:,:,selections], self.y_train)
        elif axis==1:
            return SubsetDataset(self.X_train[:,selections,:], self.y_train)

    def return_validation_dataset(self):
        return SubsetDataset(self.X_val, self.y_val)

    def return_validation_subset_dataset(self,selections,axis=2):
        if axis==2:
            return SubsetDataset(self.X_val[:,:,selections], self.y_val)
        elif axis==1:
            return SubsetDataset(self.X_val[:,selections,:], self.y_val)

    def return_y_val(self):
        return self.y_val

    def return_training_data(self):
        return [self.X_train, self.y_train]
