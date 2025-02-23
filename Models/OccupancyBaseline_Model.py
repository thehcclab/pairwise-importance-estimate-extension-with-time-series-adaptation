import torch
import logging
import os
import Models.model_func as Model_Func
import pickle
from tqdm import tqdm
import numpy as np
from typing import List
from sklearn.metrics import classification_report






class OccupancyBaseline_Model(torch.nn.Module):
    def __init__(self, device:torch.device, input_dim:List, classes:int, rnn_hidden=2, lr=0.001):
        super().__init__()

        assert len(input_dim)==2, "input_dim must have 2 dimensions"
        for i in input_dim:
            assert isinstance(i, int), "input_dim must contain int"
        assert isinstance(device, torch.device), "device is not a torch.device"

        self.device=device

        # self.rnn= torch.nn.RNN(input_dim, rnn_hidden, batch_first=True)
        self.rnn=torch.nn.LSTM(input_dim[1],rnn_hidden,batch_first=True)
        self.dropout= torch.nn.Dropout(0.2)

        self.layer_0= torch.nn.Linear(rnn_hidden, 1) # original
        self.layer_1= torch.nn.Linear(1, classes)

        # self.layer_1= torch.nn.Linear(input_dim*rnn_hidden, input_dim*rnn_hidden // 2)
        # self.layer_2= torch.nn.Linear(input_dim*rnn_hidden // 2,classes)

        self.lr= lr
        self.optimiser= None
        self.epoch=1
        self.v_dicts=[]
        self.t_dicts=[]

    def forward(self,x):
        # print("x", x.shape)
        # print("x unsqueeze", x.unsqueeze(0).shape)
        rnn_output, states= self.rnn(x)

        # print("rnn_output", rnn_output.shape)
        rnn_output= rnn_output.squeeze(0)
        rnn_output= self.dropout(rnn_output)
        # output= self.layer_2(self.layer_1(rnn_output)).softmax(dim=1)
        output= self.layer_1(self.layer_0(rnn_output)).softmax(dim=1) # original
        # print("output softmax", output.shape)

        return output


        # return self.layer_2( self.layer_1( self.layer_0(x) ) ).softmax(dim=1)
        # return self.layer_1( self.layer_0(x) ).softmax(dim=1)

    def prediction_procedure(self, dataloader, dict_flag=False):

        device= self.device

        return Model_Func.prediction_rnn(self, dataloader, device, dict_flag)

    def training_procedure(self, iteration, train_dataloader, val_dataloader, path, loss_func, optimiser, train_func, print_cycle):
        device= self.device

        path= path+"-dictionary"
        if not os.path.exists(path):
            os.makedirs(path)

        if self.optimiser==None:
            self.optimiser=optimiser

        for i in tqdm(range(iteration), desc="Iterations"):
            t_loss, t_dict= Model_Func.training(self, train_dataloader, train_func, loss_func, device)

            self.t_dicts += [t_dict]
            # pickle.dump(t_dict, open(path+"/t_dict-"+str(self.epoch)+".pkl", "wb"))

            print("Epoch ", self.epoch, ", loss", t_loss)

            if self.epoch % print_cycle==0:
                v_loss, v_dict= Model_Func.validation_rnn(self, val_dataloader, loss_func, device)
                pickle.dump(v_dict, open(path+"/v_dict-"+str(self.epoch)+".pkl","wb"))

                self.v_dicts += [v_dict]

                t_acc, v_acc= t_dict["accuracy"], v_dict["accuracy"]
                t_recall, v_recall= t_dict["macro avg"]["recall"], v_dict["macro avg"]["recall"]
                t_prec, v_prec= t_dict["macro avg"]["precision"], v_dict["macro avg"]["precision"]
                t_f1, v_f1= t_dict["macro avg"]["f1-score"], v_dict["macro avg"]["f1-score"]

#                 print("Epoch: ", i)
#                 print("t_loss: ", t_loss,", v_loss: ", v_loss)
#                 print("t_acc: ", t_acc,", v_acc: ", v_acc)
#                 print("t_recall: ", t_recall,", v_recall: ", v_recall)
#                 print("t_prec: ", t_prec, ", v_prec: ", v_prec)
#                 print("t_f: ", t_f1,", v_f: ", v_f1)
#                 print("////////")

                logging.info("Epoch: "+str(i))
                logging.info("t_loss: "+str(t_loss)+", v_loss: "+str(v_loss))
                logging.info("t_acc: "+str(t_acc)+", v_acc: "+str(v_acc))
                logging.info("t_recall: "+str(t_recall)+", v_recall: "+str(v_recall))
                logging.info("t_prec: "+str(t_prec)+", v_prec: "+str(v_prec))
                logging.info("t_f: "+str(t_f1)+", v_f: "+str(v_f1))
                logging.info("//////////")

            self.epoch += 1
