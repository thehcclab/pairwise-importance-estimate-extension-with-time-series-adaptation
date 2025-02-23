import os
import logging
import re
from utilities.Log import define_root_logger
import logging
import pickle
import torch
import utilities.Log as Log
import numpy as np
import yaml

def init(args, default_path, config):

    if not os.path.exists(default_path):
        os.makedirs(default_path)
        exp_num=0
    else:
    	dir_content=os.listdir(default_path)
    	exp_num= len( [ dir_content[i] for i in range(len(dir_content)) if re.match(pattern="exp_log[0-9]+", string=dir_content[i] ) ] )

    if args.dir=="None":
        dir_name= "exp_log"+str(exp_num) #str(exp_num+run
        os.makedirs(default_path+f"/{dir_name}");
        file_name="log"+str(exp_num)
        open(default_path+f"/{dir_name}/{file_name}.log", "w")
        define_root_logger(config["default_path"]+f"/{dir_name}/{file_name}.log")

        default_path= default_path+f"/{dir_name}/"
        logging.info("Created new directory: "+default_path)
    else:
        default_path= default_path+"/"+args.dir
        if not os.path.exists(default_path):
            os.makedirs(default_path)

        file_name="log"+args.dir[-1]

        if not os.path.exists(default_path+f"/{file_name}.log"):
            open(default_path+f"/{file_name}.log", "w")

        define_root_logger(default_path+f"/{file_name}.log")
        default_path= default_path+"/"

    return default_path

def rs_init(name, random_fold, percentile, X, x_shape=1):

    logging.info("Percentile: "+str(percentile))
    with open(f"./experiments/{name}rs{random_fold}.yaml", 'r') as stream:
        dictionary= yaml.safe_load(stream)

        array=np.array([ True for i in range(X.shape[x_shape]) ])
        for i in dictionary[percentile]:
                array[i]= False
    logging.info(f"original dim: {X.shape}")
    logging.info(f"dim {x_shape}: {np.count_nonzero(array)}" )
    # print("dim: ", X[:, array].shape)

    return array

from utilities.OccupancyDataset import OccupancyDataset
def occupancy_pre(config, default_path:str):

    logging.info("Loading Data: ...")
    data=OccupancyDataset(config=config, default_path=default_path)
    # classes=data.classes
    logging.info("Done")

    logging.info(f"X_train: {data.X_train.shape}, X_val: {data.X_val.shape}")

    train_dataloader= torch.utils.data.DataLoader(data.return_training_dataset(), batch_size=config["batch_size"])
    val_dataloader= torch.utils.data.DataLoader(data.return_validation_dataset(), batch_size=config["batch_size"])

    return train_dataloader, val_dataloader, data, data.classes

from utilities.ScikitFeatDataset import ScikitFeatDataset
def ScikitFeat_pre(args, config, default_path:str):

    logging.info("Loading Data: "+args.data+"...")

    data=ScikitFeatDataset(dataset=args.data, split=config["split"], default_path=default_path)
    # classes=data.classes
    logging.info("Done")

    logging.info(f"X_train: {len(data.train_indices)}, X_val: {len(data.val_indices)}")

    train_dataloader= torch.utils.data.DataLoader(data.return_training_dataset(), batch_size=config["batch_size"])
    val_dataloader= torch.utils.data.DataLoader(data.return_validation_dataset(), batch_size=config["batch_size"])

    return train_dataloader, val_dataloader, data, data.classes

from utilities.DryBeanDataset import DryBeanDataset
def DryBean_pre(config:dict, default_path:str):

    logging.info("Loading Data: ...")
    data=DryBeanDataset(split=config["split"], default_path=default_path)
    # classes=data.classes
    logging.info("Done")

    logging.info(f"X_train: {data.X_train.shape}, X_val: {data.X_val.shape}")

    train_dataloader= torch.utils.data.DataLoader(data.return_training_dataset(), batch_size=config["batch_size"])
    val_dataloader= torch.utils.data.DataLoader(data.return_validation_dataset(), batch_size=config["batch_size"])

    return train_dataloader, val_dataloader, data, data.classes

from utilities.MNISTDataset import MNISTDataset
def MNIST_pre(config:dict):
    logging.info("Loading Data: ...")

    dataset_train=MNISTDataset("TRAIN")
    dataset_val=MNISTDataset("TEST")
    # classes=dataset_train.classes
    logging.info("Done")

    logging.info(f"X_train: {dataset_train.X.shape}, X_val: {dataset_val.X.shape}")

    train_dataloader= torch.utils.data.DataLoader(dataset_train, batch_size=config["batch_size"])
    val_dataloader= torch.utils.data.DataLoader(dataset_val, batch_size=config["batch_size"])

    return train_dataloader, val_dataloader, dataset_val, dataset_train.classes

def simul_pre(config:dict, default_path:str):
    logging.info("Loading Data: ...")

    from utilities.SimulationStudyDataset import SimulationStudyDataset

    data=SimulationStudyDataset(split=config["split"], default_path=default_path)
    # classes=data.classes
    logging.info("Done")

    logging.info(f"X_train: {data.X_train.shape}, X_val: {data.X_val.shape}")

    train_dataloader= torch.utils.data.DataLoader(data.return_training_dataset(), batch_size=config["batch_size"])
    val_dataloader= torch.utils.data.DataLoader(data.return_validation_dataset(), batch_size=config["batch_size"])

    return train_dataloader, val_dataloader, data, data.classes

def unisimul_pre(config:dict, option:int, default_path:str):
    logging.info("Loading Data: ...")

    from utilities.UnivariateCreateDataset import UnivariateSimulationStudyDataset

    data=UnivariateSimulationStudyDataset(config=config, option=option, default_path=default_path)
    # classes=data.classes
    logging.info("Done")

    logging.info(f"X_train: {data.X_train.shape}, X_val: {data.X_val.shape}")

    train_dataloader= torch.utils.data.DataLoader(data.return_training_dataset(), batch_size=config["batch_size"])
    val_dataloader= torch.utils.data.DataLoader(data.return_validation_dataset(), batch_size=config["batch_size"])

    return train_dataloader, val_dataloader, data, data.classes
def multisimul_pre(config:dict, option:int, default_path:str):
    logging.info("Loading Data: ...")

    from utilities.MultivariateCreateDataset import MultivariateSimulationStudyDataset

    data=MultivariateSimulationStudyDataset(config=config, option=option, default_path=default_path)
    # classes=data.classes
    logging.info("Done")

    logging.info(f"X_train: {data.X_train.shape}, X_val: {data.X_val.shape}")

    train_dataloader= torch.utils.data.DataLoader(data.return_training_dataset(), batch_size=config["batch_size"])
    val_dataloader= torch.utils.data.DataLoader(data.return_validation_dataset(), batch_size=config["batch_size"])

    return train_dataloader, val_dataloader, data, data.classes

import utilities.metric_func as func
import matplotlib.pyplot as plt
def post(default_path, config, feat, model, y_val, classes, val_dataloader, name):

    print(feat)

    dir_path= default_path+f"{name}/"
    if not os.path.exists(dir_path):
             os.makedirs(dir_path)

    if not os.path.exists(dir_path+"graphs"):
        os.makedirs(dir_path+"graphs")

    figure= func.metric_graph(model.v_dicts[-1], classes=classes)
    figure.savefig(dir_path+f"graphs/{feat}-classification_report-{config['epoch']}.png")
    plt.close(figure)

    if not os.path.exists(dir_path+"dictionary"):
        os.makedirs(dir_path+"dictionary")

    prediction, v_dict=model.prediction_procedure(val_dataloader, dict_flag=True)
    pickle.dump(v_dict, open(f"{dir_path}dictionary/{feat}-v_dict-{config['epoch']}.pkl","wb"))


    ber= func.BER(prediction, y_val)
    pickle.dump(ber, open(f"{dir_path}{feat}-ber-{config['epoch']}.pkl", "wb"))

    if not os.path.exists(f"{dir_path}state_dict"):
        os.makedirs(f"{dir_path}state_dict")

    torch.save(model.state_dict(), f"{dir_path}state_dict/{feat}-state_dict-{config['epoch']}.pt")
