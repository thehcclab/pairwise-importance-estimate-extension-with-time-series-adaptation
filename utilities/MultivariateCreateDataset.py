import numpy as np
def create_dataset0(split:dict, signal_size:int, noise:float):

    def noise_maker(array,start, end, noise, signal_base=1):
        for i in range(start, end):
            array[i]= signal_base+ np.random.normal(scale=noise)
        return array

    # Variable signal strength for class, same frequency
    def create_signal(size=100, noise=0, number_of_pi=10, label=0):
        pos= round(5 *(size/100))

        signal_1= np.zeros(size)
        signal_1= noise_maker(signal_1,0, len(signal_1), noise, 1)

        signal_2= np.zeros(size)
        signal_2= noise_maker(signal_2,0, len(signal_2), noise, 1)

        signal_0= np.zeros(size)
        signal_0= noise_maker(signal_0,0, len(signal_0), noise, 1)

        signal= np.zeros(size)

        if label==0:
            signal_0= noise_maker(signal, pos*5-1, len(signal)-pos*5, noise, 2)
    #         signal_0= noise_maker(signal, 0, len(signal), noise, 2)
        elif label==1:
            signal_1= noise_maker(signal, pos*5-1, len(signal)-pos*5, noise, 2)
    #         signal_1= noise_maker(signal, 0, len(signal), noise, 2)
        elif label==2:
            signal_2= noise_maker(signal, pos*5-1, len(signal)-pos*5, noise, 2)
    #         signal_2=noise_maker(signal, 0, len(signal), noise, 2)


        return signal_0, signal_1, signal_2

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

    def noise_maker(array,start, end, noise, signal_base=1):
        for i in range(start, end):
            array[i]= signal_base+ np.random.normal(scale=noise)
        return array

    # Variable signal strength for class, same frequency
    def create_signal(size=100, noise=0, number_of_pi=10, label=0):
        pos= round(5 *(size/100))
        # Standard noise
        noise_0= np.zeros(size)
        noise_0= noise_maker(noise_0,0, len(noise_0), noise, 0)

        signal_1= np.zeros(size)
        signal_1= noise_maker(signal_1,0, len(signal_1), noise, 0)



        signal= np.zeros(size)

        if label==0:
            signal_1= noise_maker(signal, pos-1, len(signal)//4, noise, 1)
        elif label==1:
            signal_1= noise_maker(signal, len(signal)//4+pos-1, len(signal)//4*2, noise, 1)
        elif label==2:
            signal_1= noise_maker(signal, len(signal)//4*2+pos-1, len(signal)//4*3, noise, 1)

        return noise_0, signal_1

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



def create_dataset2(split:dict, signal_size:int, noise:float):

    def noise_maker(array,start, end, noise, signal_base=1):
        for i in range(start, end):
            array[i]= signal_base+ np.random.normal(scale=noise)
        return array

    # Variable signal strength for class, same frequency
    def create_signal(size=100, noise=0, number_of_pi=10, label=0):
        pos= round(5 *(size/100))
        # Standard noise
        noise_0= np.zeros(size)
        noise_0= noise_maker(noise_0,0, len(noise_0), noise, 0)

        signal_1= np.zeros(size)
        signal_1= noise_maker(signal_1,0, len(signal_1), noise, 0)

        signal_2= np.zeros(size)
        signal_2= noise_maker(signal_2,0, len(signal_2), noise, 1)

        signal= np.zeros(size)

        if label==0:
            signal_1= noise_maker(signal, pos-1, len(signal)//4, noise, 1)
        elif label==1:
            signal_1= noise_maker(signal, len(signal)//4+pos-1, len(signal)//4*2, noise, 1)
        elif label==2:
            signal_2= noise_maker(signal, len(signal)//4*2+pos-1, len(signal)//4*3, noise, 1)

        return noise_0, signal_1, signal_2

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
class MultivariateSimulationStudyDataset(Dataset):

    def __init__(self,config:dict,option:int,default_path="./experiments/multisimul_study/"): #init or load
    # assert protocol=="init" or protocol=="load", "\"load\" or \"init\""
        super().__init__()

        split= config["split"]

        if option==0:
            noise=config["noise_0"]
            dict_split=config["split_0"]
            size=config["size_0"]
            create_dataset_func= create_dataset0
        elif option==1:
            noise=config["noise_1"]
            dict_split=config["split_1"]
            size=config["size_1"]
            create_dataset_func= create_dataset1
        elif option==2:
            noise=config["noise_2"]
            dict_split=config["split_2"]
            size=config["size_2"]
            create_dataset_func= create_dataset2
        else:
            raise Exception("Invalid Option")

        try:
            self.X, self.y= pickle.load(open(f"{default_path}dataset{option}.pkl", "rb"))
        except Exception:
            self.X, self.y= create_dataset_func(dict_split, size, noise)
            self.X=np.array(self.X); self.y=np.array(self.y)
            self.X= np.transpose(self.X, axes=[0,2,1])
            pickle.dump([self.X, self.y], open(f"{default_path}dataset{option}.pkl","wb"))
        try:
            self.train_indices, self.val_indices= pickle.load(open(f"{default_path}dataset{option}-indices.pkl", "rb"))
        except FileNotFoundError:
            self.train_indices, self.val_indices= create_indices(self.y, split=split)
            pickle.dump([self.train_indices, self.val_indices], open(f"{default_path}dataset{option}-indices.pkl", "wb"))

        self.X_train= self.X[self.train_indices]; self.X_val= self.X[self.val_indices]
        self.y_train= self.y[self.train_indices]; self.y_val= self.y[self.val_indices]
        self.classes= len( np.unique(self.y) )
