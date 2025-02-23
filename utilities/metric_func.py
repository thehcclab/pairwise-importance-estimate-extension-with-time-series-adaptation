import math
import numpy as np; #import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import seaborn as sns
import pandas as pd

import torch
from scipy.io import loadmat
import pickle

def normalise(data, axis):
    minimum= data.min(axis=axis, keepdims=True)
    maximum= data.max(axis=axis, keepdims=True)

    tmp= maximum-minimum

    if np.count_nonzero(tmp==0)==0:
        return ( ( data-minimum ) / ( maximum-minimum ) )
    else:
        tmp= np.where(tmp==0, 1.0, tmp) # non-ideal
        return ( ( data-minimum ) / tmp )

# def binarise(y):
#     binary_y= y>=y.mean()
#     return binary_y.astype(int)

def select_features(idx, shape, percentile):

    weights= np.zeros(shape)

    selection= math.floor(len(weights)*0.01*int(percentile) )
    weights[ idx[:-selection] ]=1

    return weights

def BER(y_pred, y_test):
    ber=[]

    for i in np.unique(y_test):
        class_loc= np.where(y_pred==i)

        count=0
        # print("Prediction for Class",i,": ", y_test[class_loc])
        for j in y_test[class_loc]:
            if j==i:
                count += 1
        try:
            error_rate= 1-count/ len(y_pred[class_loc])
        except ZeroDivisionError:
            error_rate=1
        print("Class", i," Error Rate : ",error_rate)
        print()
        ber.append(error_rate)
    print()
    print("BER: ", np.mean(ber))

    return ber




def data_label_shuffle(data, label):

    indexes= [ i for i in range(0,len(data))]
    np.random.shuffle(indexes)
    return data[indexes], label[indexes]


def metric_percentages(dictionary, classes=2):
    prec_percentages= []
    recall_percentages= []
    f1_percentages= []

    for i in range(classes):
        try:
            prec_percentages += [dictionary[str(i)]["precision"]]
            recall_percentages += [dictionary[str(i)]["recall"]]
            f1_percentages += [dictionary[str(i)]["f1-score"]]
        except KeyError:
            prec_percentages += [0]
            recall_percentages += [0]
            f1_percentages += [0]

    return [prec_percentages, recall_percentages, f1_percentages]

def metric_graph(dictionary, classes=2):

    percentages= metric_percentages(dictionary, classes)

    f= plt.figure()

    series_dict= { "precision":percentages[0], "recall":percentages[1], "f1-score":percentages[2] }
    df= pd.DataFrame(series_dict)
    h= sns.heatmap(df.transpose(), cmap="Blues", vmin=0, vmax=1.0 )
    h.set_xlabel("Classes")
    h.set_ylabel("Metrics")
    h.set_title("Metrics for each class")
    plt.yticks(rotation=0)

    return f
