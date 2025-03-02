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
from utilities.HHAR_func import user_fold_load, user_fold_plot, HHAR_post
from utilities.script_func import init
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

feat= "NFS"

class LSTM_output(torch.nn.Module):
    def forward(self, x):
        return x[0].squeeze(0)

def main():

    fold_average=[]
    w_list=[]
    # dir_path= f"{save_path}/{feat}/"
    if not os.path.exists(save_path):
             os.makedirs(save_path)
    if not os.path.exists(f"{save_path}/weights"):
             os.makedirs(f"{save_path}/weights")
    for i in range(len(USERS)):
        logging.info(f"fold {USERS[i]}")

        logging.info("Load data")
        train_dataloader, val_dataloader, classes, input_dim = user_fold_load(i, user_fold, USERS, batch_size=config['batch_size'])
        logging.info("Done")

        baseline_model= Model.HHARBaseline_Model(device=DEVICE, input_dim=input_dim, classes=classes)
        nonlinear_func= torch.nn.Sequential(
                    torch.nn.LSTM(input_dim,input_dim),
                    LSTM_output(),
                    torch.nn.Sigmoid()
                        ).to(DEVICE)

        model= Models.MultivariateSeries_NeuralFS(device=DEVICE, input_dim=input_dim,
                                                nonlinear_func=nonlinear_func,decision_net=baseline_model).to(DEVICE)

        loss = torch.nn.CrossEntropyLoss()
        optimiser= torch.optim.RMSprop(model.parameters(), lr=config['learning_rate'])
        train_func= Model_Func.train_rnn

        model.training_procedure(iteration=EPOCH, train_dataloader=train_dataloader, val_dataloader=val_dataloader, print_cycle=5,path=f"{save_path}/dictionary/intermdiate_dicts", loss_func=loss, optimiser=optimiser, train_func=train_func)
        pickle.dump( model.return_pairwise_weights(), open(f"{save_path}/weights/{fold_name}-{feat}-fold{USERS[i]}-w-e{EPOCH}.pkl", "wb") )
        prediction, dictionary= model.prediction_procedure(val_dataloader, dict_flag=True)

        dict_path= f"{save_path}/dictionary"
        state_path= f"{save_path}/state_dict"
        HHAR_post(dict_path, state_path, USERS[i], dictionary, model, f"{fold_name}-{feat}", EPOCH)

        fold_average.append(dictionary)
        w_list.append(model.return_pairwise_weights())

    pickle.dump(fold_average, open(f"{save_path}/{fold_name}-avg_dict-e{EPOCH}.pkl", 'wb'))
    pickle.dump(np.array(w_list).mean(axis=0), open(f"{save_path}/{fold_name}-avg_w-e{EPOCH}.pkl","wb"))
    return fold_average

if __name__=="__main__":
    parser= argparse.ArgumentParser(description='This script is used to produce the baselines for the Neural Feature Selection method with the Element-wise Multiplication adaptation, and stores the training-validation split of the data for later use.')
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

    save_path= f"{default_path}{args.device_sensor}/{feature_setting}/{feat}"

    fold_average= main()
    user_fold_plot(save_path, fold_average, f"{fold_name}-e{EPOCH}", USERS, True)
    logging.info(f"EPOCH {EPOCH}")
    tmp=[]
    for i, dictionary in enumerate(fold_average):
        logging.info(f"fold {USERS[i]} {dictionary['weighted avg']['f1-score']}")
        tmp.append(dictionary['weighted avg']['f1-score'])

    logging.info(f"average {np.mean(tmp)}")
