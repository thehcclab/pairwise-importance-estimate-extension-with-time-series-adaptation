import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
import os
import argparse
from typing import List
import sys
import Models.HHARBaseline_Model as Model
import Models.model_func as Model_Func
import Models.multi_models as Models
from utilities.script_func import init
from utilities.subset_func import Grad_STD_with_grad, Grad_AUC_with_grad, Grad_ROC_with_grad, grad_average_ranking
from utilities.HHAR_func import user_fold_load, user_fold_plot, HHAR_post
from utilities.Log import define_root_logger
import logging
logging.info("")

USERS= ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
import yaml, io
with open("./config/HHAR_config.yaml", 'r') as stream:
    config= yaml.safe_load(stream)

    config["default_path"]= str(config["default_path"])
    config["cuda"]= str(config["cuda"])
    config["epoch"]= int(config["epoch"])
    config["batch_size"]=int(config["batch_size"])
    config["data_saved_path"]= str(config["data_saved_path"])

import os
os.environ["CUDA_VISIBLE_DEVICES"]=config["cuda"]

default_path=config["default_path"]
EPOCH=config["epoch"]
DEVICE= torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

feat= "multivar-Grad"

def main():

    fold_average=[]
    # gradstd_list=[]
    # gradauc_list=[]
    # gradroc_list=[]
    # dir_path= f"{save_path}/{feat}/"
    if not os.path.exists(save_path):
             os.makedirs(save_path)
    if not os.path.exists(f"{save_path}/list"):
             os.makedirs(f"{save_path}/list")
    for i in range(len(USERS)):
        logging.info(f"fold {USERS[i]} {user_fold[USERS[i]][0][0].shape}")

        logging.info("Load data")
        train_dataloader, val_dataloader, classes, input_dim = user_fold_load(i, user_fold, USERS, batch_size=config['batch_size'])
        logging.info("Done")

        win_len=0
        for t in train_dataloader:
            win_len= t[0].shape[1]
            break
        print("win_len",win_len, "input dim",input_dim, "train_dataloader", len(train_dataloader), "val_dataloader", len(val_dataloader))

        baseline_model= Model.HHARBaseline_Model(device=DEVICE, input_dim=input_dim, classes=classes)
        model= Models.Multivariate_IEGradient(device=DEVICE, input_dim=[win_len,input_dim],
                                  model=baseline_model, save_path=f"{save_path}/list", fold=USERS[i], fold_name=fold_name).to(DEVICE)

        loss = torch.nn.CrossEntropyLoss()
        optimiser= torch.optim.RMSprop(baseline_model.parameters(), lr=config['learning_rate'])
        train_func= Model_Func.grad_train_rnn_alt

        # model.load_state_dict(torch.load(f"{save_path}/state_dict/{fold_name}-multivar-Grad-state_dict-fold{USERS[i]}-e{EPOCH}.pt"))

        model.training_procedure(iteration=EPOCH, train_dataloader=train_dataloader, val_dataloader=val_dataloader, print_cycle=5,path=f"{save_path}/dictionary/intermdiate_dicts", loss_func=loss, optimiser=optimiser, train_func=train_func)

        prediction, dictionary= model.prediction_procedure(val_dataloader, dict_flag=True)

        dict_path= f"{save_path}/dictionary"
        state_path= f"{save_path}/state_dict"
        HHAR_post(dict_path, state_path, USERS[i], dictionary, model, f"{fold_name}-{feat}", EPOCH)

        fold_average.append(dictionary)

        ie_grad=[]
        for e in range(1,EPOCH+1):
            g= pickle.load(open(f"{save_path}/list/fold{USERS[i]}/{fold_name}-multivar-grads-e{e}.pkl", "rb"))
            ie_grad.append(np.array(g).sum(axis=0))
    #         ie_grad.append(np.array(g).std(axis=0))
        pickle.dump( np.array(ie_grad), open(f"{save_path}/list/{fold_name}-fold{USERS[i]}-multivar-epoch_sum-e{EPOCH}.pkl", "wb") )

        ie_grad=[]
        for e in range(1,EPOCH+1):
            g= pickle.load(open(f"{save_path}/list/fold{USERS[i]}/{fold_name}-multivar-grads-e{e}.pkl", "rb"))
            ie_grad.append(np.array(g).std(axis=0))
    #         ie_grad.append(np.array(g).std(axis=0))
        pickle.dump( np.array(ie_grad), open(f"{save_path}/list/{fold_name}-fold{USERS[i]}-multivar-epoch_std-e{EPOCH}.pkl", "wb") )


    pickle.dump(fold_average, open(f"{save_path}/{fold_name}-multivar-avg_dict-e{EPOCH}.pkl", 'wb'))

    return fold_average

if __name__=="__main__":
    parser= argparse.ArgumentParser(description='This script is used to produce the baselines for PIEE\'s Gradient based Aggregated Hadamard product method, and stores the training-validation split of the data for later use.')
    parser.add_argument('-dir', help='default path of where experiment is saved. Default: \"exp_log0\"', default="None")
    parser.add_argument("-device_sensor", help="Corresponding device and sensor's csv to be used. Default: 'phone_accel'", default="phone_accel")
    args= parser.parse_args()
    args.dir= str(args.dir)
    args.device_sensor= str(args.device_sensor)

    default_path= init(args, default_path, config)

    fold_name="userfold"
    feature_setting= "original-minmax"
    fold_setting= f"{fold_name}-{feature_setting}"
    logging.info("Load fold dictionary...")
    try:
        user_fold= pickle.load(open(f"{config['data_saved_path']}/{args.device_sensor}/{fold_setting}.pkl", "rb"))
    except:
        print(f"pickle file does not exist at {config['data_saved_path']}/{args.device_sensor}/{fold_setting}.pkl. Use HHAR_extract_measurement.py from the utilities folder.")
        sys.exit()

    logging.info("Done")

    save_path= f"{default_path}{args.device_sensor}/{feature_setting}/Grad"

    fold_average= main()
    user_fold_plot(save_path, fold_average, f"{fold_name}-multivar-e{EPOCH}", USERS,True)
    logging.info(f"EPOCH {EPOCH}")
    tmp=[]
    for i, dictionary in enumerate(fold_average):
        logging.info(f"fold {USERS[i]} {dictionary['weighted avg']['f1-score']}")
        tmp.append(dictionary['weighted avg']['f1-score'])


    logging.info(f"average {np.mean(tmp)}")
