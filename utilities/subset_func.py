import os
import pickle
import matplotlib.pyplot as plt
import torch
import numpy as np
import utilities.metric_func as func
import math
import logging

def process_cov(array, threshold, channels, dictionary={}, transpose=True):

    for i in channels:
        dictionary[i]=0

    i=0; j=0;
    print("Threshold: ", threshold)
    print()

    if transpose:
        array= array.T

    for row in np.where( abs(np.cov(array)) > threshold, np.cov(array), 0):
        for column in row:
#             print("j", j, "i", i)
#             print("skip_condition", skip_condition)
#             print()
            if j==i:
                break

            if abs(column) > threshold and column != 1:
                # print(f"cov: {round(column,3)}, {channels[i]} {channels[j]}")
                dictionary[channels[i]] += abs(column)
                dictionary[channels[j]] += abs(column)
            j+=1
        i+=1
        j=0

    return dictionary

def return_epoch_stat(grad_all, stat="sum"):
    epochs= []

    grad_all= np.nan_to_num(grad_all) # NaN conversion

    if stat=="sum":
        for epoch in grad_all:
            epochs.append( epoch.sum(axis=0) )
    elif stat=="std":
        for epoch in grad_all:
            epochs.append( epoch.std(axis=0) )
    elif stat=="mean":
            for epoch in grad_all:
                epochs.append( epoch.mean(axis=0) )

    return np.array(epochs)

def clean(grad_all):
    grad_all= np.nan_to_num(grad_all) # NaN conversion
    grad_all= np.where(grad_all==-np.inf,0,grad_all) # np.inf converseion
    grad_all= np.where(grad_all==np.inf,0,grad_all) # np.inf converseion

    return grad_all

# def scale(grad_all, scaler):
#     for i in range(grad_all.shape[2]):
#             grad_all[:,:,i]= scaler.fit_transform(grad_all[:,:,i])

# #     for i in range(len(grad_all)):
# #         grad_all[i,:,:]= scaler.fit_transform(grad_all[i,:,:])
#     return grad_all

def threshold(grad_all, upper_quantile=95, lower_quantile=5):

    quantile_list= np.percentile(grad_all, [upper_quantile, lower_quantile])

    grad_upper_quantile= quantile_list[0]
    grad_lower_quantile= quantile_list[1]

    grad_all= np.where(grad_all>grad_upper_quantile,grad_upper_quantile,grad_all) # threshold
    grad_all= np.where(grad_all<grad_lower_quantile,grad_lower_quantile,grad_all)

    return grad_all

def dictionary_sort(x):
        return x[1]

def alternate_rank(dictionary, reverse=True):
    """When np.percentile is 0"""

    sorted_array= sorted(dictionary.items(), key=dictionary_sort, reverse=reverse)
    index= [idx for idx, _ in sorted_array]

    return index

def grad_average_ranking(input_dim, grad_list):
        """
        low value in avg_ranking means higher importance
        each position in avg_ranking corresponds to the relative postioning of sensors
        """
        dictionary={}
        for i in range(input_dim):
            dictionary[i]=[]

        for ranking_list in grad_list:
            for idx, channel_idx in enumerate(ranking_list):
                dictionary[channel_idx].append(idx)

        avg_ranking=[]
        for i in range(input_dim):
            avg_ranking.append(np.array(dictionary[i]).mean())
        return avg_ranking

def select_features(idx, shape, percentile):
    """binarise the weights and select according to percentile"""
    weights= np.zeros(shape, dtype=bool)

    selection= math.floor(len(weights)*0.01*int(percentile) )
    weights[ idx[:-selection] ]=True

    logging.info(f"selected {len(idx[:-selection])} / {len(weights)}")

    return weights


def Grad_AUC(load_w, epoch, subset, X, x_shape=1):
        percentile= int(subset) # // 2 will make it 5% instead
        channels= np.arange(X.shape[x_shape])

        grads= pickle.load(open(load_w, "rb"))
        grad_all= np.array(grads).reshape(epoch,-1,X.shape[x_shape])

        epoch_sum= return_epoch_stat(grad_all, stat="sum")
        dictionary={}
        for idx, name in zip(range(len(channels)), channels):
            dictionary[name]= np.trapz(abs(epoch_sum[:, idx]))

        weights= np.array(list(dictionary.values()))
        q= np.percentile(list(dictionary.values()), [percentile])

        if q==0.:
            weights= select_features(alternate_rank(dictionary), X.shape[x_shape], subset)
        else:
            weights= weights > q

        return weights

def Grad_AUC_with_grad(grads, epoch, X,x_shape=1):

    channels=np.arange(X.shape[x_shape])
    grad_all= np.array(grads).reshape(epoch, -1,X.shape[x_shape])

    epoch_sum= return_epoch_stat(grad_all, stat="sum")
    dictionary={}
    for idx, name in zip(range(len(channels)), channels):
        dictionary[name]= np.trapz(abs(epoch_sum[:,idx]))

    return alternate_rank(dictionary, reverse=False)

def Grad_AUC_with_multivar_grad(load_w, epoch, subset, X, x_shape=1,scaling=True):
    percentile= int(subset)
    channels=np.arange(X.shape[x_shape+1])

    grads= pickle.load(open(load_w, "rb"))
    grad_all= np.array(grads).reshape(epoch,-1,X.shape[x_shape], X.shape[x_shape+1])

    if scaling:
        scaler= MinMaxScaler()

        epoch= return_epoch_stat(grad_all, "sum")
        dictionary={}
        for idx, name in zip(range(len(channels)), channels):
            dictionary[name]= np.trapz(abs(epoch[:, :, idx]), axis=0)
            scaler.partial_fit(dictionary[name].reshape(-1,1))


        for key in dictionary.keys():
            dictionary[key]= scaler.transform(dictionary[key].reshape(-1,1)).reshape(-1)
    else:
        epoch= return_epoch_stat(grad_all, "sum")
        dictionary={}
        for idx, name in zip(range(len(channels)), channels):
            dictionary[name]= np.trapz(abs(epoch[:, :, idx]), axis=0)

    weights= np.array(list(dictionary.values()))
    q= np.percentile(list(dictionary.values()), [percentile])


    weights= weights > q

    return weights


def Grad_ROC(load_w, epoch, subset, X, x_shape=1):
        percentile= int(subset) #// 2
        channels= np.arange(X.shape[x_shape])

        grads= pickle.load(open(load_w, "rb"))
        grad_all= np.array(grads).reshape(epoch,-1,X.shape[x_shape])

        epoch_sum= return_epoch_stat(grad_all, stat="sum")
        dictionary={}
        for i in range(len(channels)):
            rolled= np.roll( epoch_sum[:, i], 1 )
            rolled[0]=0.
            diff= epoch_sum[:, i]-rolled
            dictionary[channels[i]]= np.trapz(abs(diff))

        weights= np.array(list(dictionary.values()))
        q= np.percentile(list(dictionary.values()), [percentile])

        if q==0.:
            weights= select_features(alternate_rank(dictionary), X.shape[x_shape], subset)
        else:
            weights= weights > q

        return weights

def Grad_ROC_with_grad(grads, epoch, X, x_shape=1):
    channels=np.arange(X.shape[x_shape])
    grad_all= np.array(grads).reshape(epoch, -1,X.shape[x_shape])

    epoch_sum= return_epoch_stat(grad_all, stat="sum")
    dictionary={}
    for i in range(len(channels)):
        rolled= np.roll( epoch_sum[:, i], 1 )
        rolled[0]=0.
        diff= epoch_sum[:, i]-rolled
        dictionary[channels[i]]= np.trapz(abs(diff))

    return alternate_rank(dictionary, reverse=False)

def Grad_ROC_with_multivar_grad(load_w, epoch, subset, X, x_shape=1, scaling=True):

    channels=np.arange(X.shape[x_shape+1])
    percentile= int(subset)
    grads= pickle.load(open(load_w, "rb"))
    grad_all= np.array(grads).reshape(epoch,-1,X.shape[x_shape], X.shape[x_shape+1])

    if scaling:
        scaler= MinMaxScaler()

        epoch= return_epoch_stat(grad_all, "sum")
        dictionary={}
        for idx, name in zip(range(len(channels)), channels):
            rolled= np.roll( epoch[:,:, idx], 1 )
            rolled[0]=0.
            diff= epoch[:,:, idx]-rolled
            dictionary[channels[idx]]= np.trapz(abs(diff), axis=0)
#             scaler= MinMaxScaler()
#             dictionary[channels[idx]]=
            scaler.partial_fit(dictionary[name].reshape(-1,1)) #.reshape(-1)


        for key in dictionary.keys():
            dictionary[key]= scaler.transform(dictionary[key].reshape(-1,1)).reshape(-1).sum()
    else:
        epoch= return_epoch_stat(grad_all, "sum")
        dictionary={}
        for idx, name in zip(range(len(channels)), channels):
            rolled= np.roll( epoch[:,:, idx], 1 )
            rolled[0]=0.
            diff= epoch[:,:, idx]-rolled
            dictionary[channels[idx]]= np.trapz(abs(diff), axis=0).sum()


    weights= np.array(list(dictionary.values()))
    q= np.percentile(list(dictionary.values()), [percentile])


    weights= weights > q

    return weights


def Grad_COV(load_w, epoch, subset, X, x_shape=1):
        percentile= int(subset) #// 2
        channels= np.arange(X.shape[x_shape])

        grads= pickle.load(open(load_w, "rb"))
        grad_all= np.array(grads).reshape(epoch,-1,X.shape[x_shape])

        epoch_sum= return_epoch_stat(grad_all, stat="sum")
        dictionary= process_cov(epoch_sum, abs(np.cov(epoch_sum.T )).mean(), channels)

        weights= np.array(list(dictionary.values()))
        q= np.percentile(list(dictionary.values()), [percentile])

        if q==0.:
            weights= select_features(alternate_rank(dictionary), X.shape[x_shape], subset)
        else:
            weights= weights > q

        return weights

def Grad_STD(load_w, epoch, subset, X, x_shape=1):
        percentile= int(subset) #// 2
        channels= np.arange(X.shape[x_shape])

        grads= pickle.load(open(load_w, "rb"))
        grad_all= np.array(grads).reshape(epoch,-1,X.shape[x_shape])

        epoch_std= return_epoch_stat(grad_all, stat="std")
        dictionary={}
        for i, j in zip(range(len(channels)), channels):
            dictionary[j]=np.trapz(epoch_std[:, i])# AuC

        weights= np.array(list(dictionary.values()))
        q= np.percentile(list(dictionary.values()), [percentile])

        if q==0.:
            weights= select_features(alternate_rank(dictionary), X.shape[x_shape], subset)
        else:
            weights= weights > q

        return weights

def Grad_STD_with_grad(load_w, epoch, X, x_shape=1):
        channels= np.arange(X.shape[x_shape])

        grads= pickle.load(open(load_w, "rb"))
        grad_all= np.array(grads).reshape(epoch,-1,X.shape[x_shape])

        epoch_std= return_epoch_stat(grad_all, stat="std")
        dictionary={}
        for i, j in zip(range(len(channels)), channels):
            dictionary[j]=np.trapz(epoch_std[:, i])# AuC

        return alternate_rank(dictionary, reverse=False)

def Grad_STD_with_multivar_grad(load_w, epoch, subset, X, x_shape=1,scaling=True):
    percentile= int(subset)
    channels=np.arange(X.shape[x_shape+1])

    grads= pickle.load(open(load_w, "rb"))
    grad_all= np.array(grads).reshape(epoch,-1,X.shape[x_shape], X.shape[x_shape+1])

    if scaling:
        scaler= MinMaxScaler()

        epoch= return_epoch_stat(grad_all, "std")
        dictionary={}
        for idx, name in zip(range(len(channels)), channels):
            dictionary[name]= np.trapz(abs(epoch[:, :, idx]), axis=0)
            scaler.partial_fit(dictionary[name].reshape(-1,1))


        for key in dictionary.keys():
            dictionary[key]= scaler.transform(dictionary[key].reshape(-1,1)).reshape(-1).sum()
    else:
        epoch= return_epoch_stat(grad_all, "std")
        dictionary={}
        for idx, name in zip(range(len(channels)), channels):
            dictionary[name]= np.trapz(abs(epoch[:, :, idx]), axis=0).sum()


    weights= np.array(list(dictionary.values()))
    q= np.percentile(list(dictionary.values()), [percentile])

    weights= weights > q

    return weights




from scikit_feature.skfeature.function.similarity_based import fisher_score
def Fisher_Score(default_path, load_w, model, subset, X_shape, X, y):

        try:
            idx= pickle.load(open(load_w, "rb"))
        except FileNotFoundError:
            score= fisher_score.fisher_score(X, y)
    #        score= fisher_score.fisher_score(X_train, y_train)
            idx= fisher_score.feature_ranking(score)

            if not os.path.exists(default_path+model):
                os.makedirs(default_path+model)

            pickle.dump(idx, open(load_w, "wb"))

    #     weights= np.zeros(X_train.shape[1])

    #     selection= math.floor(len(weights)*0.01*int(args.subset) )
    #     weights[ idx[:-selection] ]=1

        weights= select_features(idx, X_shape, subset)
        return weights

from scikit_feature.skfeature.function.statistical_based import f_score
def FScore(default_path, load_w, model, subset, X_shape, X, y):

    try:
        idx= pickle.load(open(load_w, "rb"))
    except FileNotFoundError:
        score= f_score.f_score(X, y)
        idx= f_score.feature_ranking(score)
        if not os.path.exists(default_path+model):
            os.makedirs(default_path+model)

        pickle.dump(idx, open(load_w, "wb"))

    weights= select_features(idx, X_shape, subset)

    return weights

def Percentile_Subset(load_w, subset):

    weights= np.array( pickle.load(open(load_w, "rb")) )
    q= np.percentile(weights, [int(subset)])
    weights= weights > q

    logging.info(f"Subset {np.count_nonzero(weights)}/ {len(weights)}")

    return weights

def Multivar_Percentile_Subset(load_w, subset, axis=0):

    weights= np.array( pickle.load(open(load_w, "rb")) )
    weights= weights.sum(axis=axis)
    q= np.percentile(weights, [int(subset)])
    # weights= np.where(weights<=q,0,weights)
    weights= weights > q

    logging.info(f"Subset {np.count_nonzero(weights)}/ {len(weights.reshape(-1))}")

    return weights#np.where(weights<=q,0,weights)

def subset_post(default_path, config, feat, model, y_val, classes, val_dataloader, subset_path):
    if not os.path.exists(f"{subset_path}/subset"):
        os.makedirs(f"{subset_path}/subset")

    if not os.path.exists(f"{subset_path}/subset/"+"graphs"):
        os.makedirs(f"{subset_path}/subset/"+"graphs")

    figure= func.metric_graph(model.v_dicts[-1], classes=classes)
    figure.savefig(f"{subset_path}/subset/graphs/{feat}-classification_report-"+str(config["epoch"])+".png")
    plt.close(figure)

    if not os.path.exists(f"{subset_path}/subset/dictionary"):
        os.makedirs(f"{subset_path}/subset/dictionary")

    prediction, v_dict=model.prediction_procedure(val_dataloader, dict_flag=True)
    pickle.dump(v_dict, open(f"{subset_path}/subset/dictionary/{feat}-v_dict-{config['epoch']}.pkl","wb"))

    ber= func.BER(prediction, y_val)
    pickle.dump(ber, open(f"{subset_path}/subset/{feat}-ber-{config['epoch']}.pkl", "wb"))

    if not os.path.exists(f"{subset_path}/state_dict/subset"):
        os.makedirs(f"{subset_path}/state_dict/subset")

    torch.save(model.state_dict(), f"{subset_path}/state_dict/subset/{feat}-state_dict-{config['epoch']}.pt")
