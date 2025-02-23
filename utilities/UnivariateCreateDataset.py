import numpy as np
def create_dataset0(split:dict, signal_size:int, noise:float):

    def noise_maker(array,start, end, noise):
        for i in range(start, end):
            array[i]= 1+ np.random.normal(scale=noise)
        return array

    def create_signal(size=100, label=0, noise=0):
        assert size % 10 == 0, "Needs to be a factor of 10"

        position= round(size/5)
        scale= round(5 *(size/100))
        array= np.zeros(size)

        if noise:
            for i in range(size):
                array[i] += np.random.normal(scale=noise)
        else:
            array[scale-1:position+scale]=1
            array[position*4-1:size-scale]=1

            array[position*2-1:position*2+scale]=1
            array[position*3-1:position*3+scale]=1

        if label==0:
            array= noise_maker(array,scale-1,position+scale, noise)
            return array
        elif label==1:
            if noise:
                array= noise_maker(array,position*2-1, position*2+2*scale, noise)
            else:
                array[position*2-1:position*2+2*scale]=1

            return array
        elif label==2:
            if noise:
                array= noise_maker(array,position*3-1,position*3+2*scale, noise)
            else:
                array[position*3-1:position*3+2*scale]=1
            return array

    dataset_X=[]
    dataset_y=[]

    for i in range(split[0]):
        dataset_X.append(create_signal(size=signal_size, label=0, noise=noise))
        dataset_y.append(0)

    for i in range(split[1]):
        dataset_X.append(create_signal(size=signal_size, label=1, noise=noise))
        dataset_y.append(1)

    for i in range(split[2]):
        dataset_X.append(create_signal(size=signal_size, label=2, noise=noise))
        dataset_y.append(2)

    return dataset_X, dataset_y

def create_dataset1(split:dict, signal_size:int, noise:float):

    def noise_maker(array,start, end, noise):
        for i in range(start, end):
            array[i]= 1+ np.random.normal(scale=noise)
        return array

    def create_signal(size=100, label=0, noise=0):
        assert size % 10 == 0, "Needs to be a factor of 10"

        position= round(size/5)
        scale= round(5 *(size/100))
        array= np.zeros(size)

        if noise:

            for i in range(size):
                array[i] += np.random.normal(scale=noise)

            array= noise_maker(array,scale-1,position+scale, noise)
            array= noise_maker(array,position*4-1,size-scale, noise)
            array= noise_maker(array,position*2-1,position*2+scale, noise)
            array= noise_maker(array,position*3-1,position*3+scale, noise)
        else:
            array[scale-1:position+scale]=1
            array[position*4-1:size-scale]=1

            array[position*2-1:position*2+scale]=1
            array[position*3-1:position*3+scale]=1

        if label==0:
            return array
        elif label==1:
            if noise:
                array= noise_maker(array,position*2-1, position*2+2*scale, noise)
            else:
                array[position*2-1:position*2+2*scale]=1

            return array
        elif label==2:
            if noise:
                array= noise_maker(array,position*3-1,position*3+2*scale, noise)
            else:
                array[position*3-1:position*3+2*scale]=1
            return array


    dataset_X=[]
    dataset_y=[]

    for i in range(split[0]):
        dataset_X.append(create_signal(size=signal_size, label=0, noise=noise))
        dataset_y.append(0)

    for i in range(split[1]):
        dataset_X.append(create_signal(size=signal_size, label=1, noise=noise))
        dataset_y.append(1)

    for i in range(split[2]):
        dataset_X.append(create_signal(size=signal_size, label=2, noise=noise))
        dataset_y.append(2)

    return dataset_X, dataset_y


from utilities.Dataset import *
from utilities.preproc_func import *
import pickle
class UnivariateSimulationStudyDataset(Dataset):

    def __init__(self,config:dict,option:int,default_path="./experiments/unisimul_study/"): #init or load
        # assert protocol=="init" or protocol=="load", "\"load\" or \"init\""
        super().__init__()

        split= config["split"]

        if option==0:
            noise=config["noise_0"]
            size=config["size_0"]
            split_dict= config["split_0"]
            create_dataset_func= create_dataset0
        elif option==1:
            noise=config["noise_1"]
            size=config["size_1"]
            split_dict= config["split_1"]
            create_dataset_func= create_dataset1
        else:
            raise Exception("Invalid Option")

        try:
            self.X, self.y= pickle.load(open(f"{default_path}dataset{option}.pkl", "rb"))
        except Exception:
            self.X, self.y= create_dataset_func(split_dict, size, noise)
            self.X=np.array(self.X);self.y=np.array(self.y)
            pickle.dump([self.X, self.y], open(f"{default_path}dataset{option}.pkl","wb"))

        try:
            self.train_indices, self.val_indices= pickle.load(open(f"{default_path}dataset{option}-indices.pkl", "rb"))
        except FileNotFoundError:
            self.train_indices, self.val_indices= create_indices(self.y, split=split)
            pickle.dump([self.train_indices, self.val_indices], open(f"{default_path}dataset{option}-indices.pkl", "wb"))

        self.X_train= self.X[self.train_indices]; self.X_val= self.X[self.val_indices]
        self.y_train= self.y[self.train_indices]; self.y_val= self.y[self.val_indices]
        self.classes= len( np.unique(self.y) )
