import matplotlib
matplotlib.use("Agg")

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import yaml, io

with open("./config/occupancy_config.yaml", 'r') as stream:
        config= yaml.safe_load(stream)

config["default_path"]= str(config["default_path"])
config["split"]= float(config["split"])
config["cuda"]= str(config["cuda"])
config["epoch"]= int(config["epoch"])
config["batch_size"]=int(config["batch_size"])

import os
os.environ["CUDA_VISIBLE_DEVICES"]=config["cuda"]

default_path=config["default_path"]

from utilities.script_func import init, post, occupancy_pre
import pickle
import logging
logging.info("")

def main():
    import Models.multi_models as Models
    import Models.model_func as Model_Func
    from Models.OccupancyBaseline_Model import OccupancyBaseline_Model

    input_dim=data.X.shape[1:]
    baseline_model= OccupancyBaseline_Model(device=device, input_dim=input_dim, classes=classes).to(device)
    class LSTM_output(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x[0].squeeze(0)

    nonlinear_func= torch.nn.Sequential(
            torch.nn.LSTM(data.X.shape[2],data.X.shape[2]),
            LSTM_output(),
            torch.nn.Sigmoid()
                ).to(device)

    model= Models.MultivariateSeries_NeuralFS(device=device, input_dim=data.X.shape[2],
                                      nonlinear_func=nonlinear_func,decision_net=baseline_model).to(device)
    loss = torch.nn.CrossEntropyLoss()
    optimiser= torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # for i in model.parameters():
    #     print(i.shape)
    model.training_procedure(iteration=config["epoch"], train_dataloader=train_dataloader, val_dataloader=val_dataloader, print_cycle=config["print_cycle"],path=default_path+"dictionary/"+feat, loss_func=loss, optimiser=optimiser, train_func=Model_Func.train_rnn)
    dir_path= default_path+"NFS/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    pickle.dump( model.return_pairwise_weights(), open(dir_path+feat+f"-w-{config['epoch']}.pkl", "wb") )
    return model

if __name__=="__main__":

    import argparse

    parser= argparse.ArgumentParser(description='This script is used to produce the baselines for the Neural Feature Selection method with the Element-wise Multiplication adaptation, and stores the training-validation split of the data for later use.')

    parser.add_argument('-dir', help='default path of where experiment is saved. Default: \"exp_log0\"', default="None")
    parser.add_argument('-opt', help="option for dataset to use or create", default=0)
    args= parser.parse_args()
    args.dir= str(args.dir)
    args.opt= int(args.opt)

    feat= f"occupancy-NFS"

    default_path= init(args, default_path, config)
    train_dataloader, val_dataloader, data, classes=  occupancy_pre(config, default_path)
    model= main()
    post(default_path, config, feat, model, data.y_val.reshape(-1), classes, val_dataloader, "NFS")
