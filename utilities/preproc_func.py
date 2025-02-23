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
    """normalise data by the axes"""
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

def create_indices(Y, split=0.4):
    """splitting each class approximately equally"""
    split= 1- split

    dictionary_label={}
    for i in range(len(np.unique(Y))):
        dictionary_label[i]= []

    count=0
    for i in np.reshape(Y, -1):
        dictionary_label[i].append(count)
        count += 1

    for i in dictionary_label.keys():
        np.random.shuffle(dictionary_label[i])
        print(len(dictionary_label[i]))

    train_indices=[]
    val_indices=[]
    for i, j in dictionary_label.items():
#         print(i, ":", len(j))
        train_indices.append(j[:math.ceil(len(j) * split)])
        val_indices.append(j[math.ceil(len(j) * split):])
#         print( math.ceil(len(j) *0.6), math.floor(len(j) * 0.4) )
#         print()

    train_indices= np.concatenate(train_indices)
    val_indices= np.concatenate(val_indices)

    return train_indices, val_indices

def convert_labels(labels):
    """Converting labels for any ys"""
    old_labels= np.unique(labels)
    new_labels= [i for i in range(len(np.unique(labels)))]

    for old_label, new_label in zip(old_labels, new_labels):
        labels[labels==old_label]= new_label

    return labels, len(old_labels)


def data_label_shuffle(data, label):
    """shuffle data through indices"""
    indices= [ i for i in range(0,len(data))]
    np.random.shuffle(indices)
    return data[indices], label[indices]
