import torch
import os
from tqdm import tqdm
import pickle
import Models.model_func as Model_Func
import logging
import functools
from typing import List

class SimulationStudyBaseline_Model(torch.nn.Module):

    def __init__(self, device:torch.device, input_dim=10, classes=2, lr=0.001):
        super().__init__()

        assert isinstance(input_dim, int), "input_dim is not an int"
        assert isinstance(device, torch.device), "device is not a torch.device"

        self.device=device

#         self.layer_0= torch.nn.Linear(input_dim, 33) # 2/8 fault
#         self.layer_1= torch.nn.Linear(33,2)
#         self.layer_2= torch.nn.Linear(2,classes)

        # self.layer_0= torch.nn.Linear(input_dim, 174) # 0/5 0.89 <--- this is the chosen one
        # self.layer_1= torch.nn.Linear(174, classes)

        self.layer_0= torch.nn.Linear(input_dim, 160) # 0/5 0.88-0.9
        self.layer_1= torch.nn.Linear(160, 44)
        self.layer_2= torch.nn.Linear(44,25)
        self.layer_3= torch.nn.Linear(25, 166)
        self.layer_4= torch.nn.Linear(166, classes)

#         self.layer_0= torch.nn.Linear(input_dim, 29)# 2 0.86
#         self.layer_1= torch.nn.Linear(29, 31)
#         self.layer_2= torch.nn.Linear(31,classes)

        self.relu= torch.nn.ReLU()

        self.epoch=1
        self.v_dicts=[]
        self.t_dicts=[]
        self.optimiser= None
        self.lr= lr

    def forward(self,x):
#         return self.layer_2( self.relu( self.layer_1( self.layer_0(x).sigmoid() ) ) ).softmax(dim=1)
        # return self.layer_1( self.layer_0(x).sigmoid() ).softmax(dim=1)
        return self.layer_4(  self.relu( self.layer_3( self.layer_2( self.relu( self.layer_1( self.relu( self.layer_0(x) ) ) ) ).sigmoid() ) )  ).softmax(dim=1)
#         return self.layer_2( self.layer_1( self.layer_0(x).sigmoid() ).sigmoid() ).softmax(dim=1)

    def set_optimiser(self, optimiser):
        self.optimiser= optimiser

    def prediction_procedure(self, dataloader, dict_flag=False):

        device= self.device

        return Model_Func.prediction(self, dataloader, device, dict_flag)

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

            print("Epoch ", i, ", loss", t_loss)

            if i % print_cycle==0:
                v_loss, v_dict= Model_Func.validation(self, val_dataloader, loss_func, device)
                # pickle.dump(v_dict, open(path+"/v_dict-"+str(self.epoch)+".pkl","wb"))

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

class UnivariateSimulationStudyBaseline_Model(torch.nn.Module):

    def __init__(self, device:torch.device, input_dim:int, classes=3, lr=0.001):
        super().__init__()

        assert isinstance(input_dim, int), "input_dim is not an int"
        assert isinstance(device, torch.device), "device is not a torch.device"

        self.device=device
        self.layer_0= torch.nn.Linear(input_dim, classes)

        self.epoch=1
        self.v_dicts=[]
        self.t_dicts=[]
        self.optimiser= None
        self.lr= lr

    def forward(self,x):
        return self.layer_0(x).softmax(dim=1)

    def prediction_procedure(self, dataloader, dict_flag=False):

        device= self.device

        return Model_Func.prediction(self, dataloader, device, dict_flag)

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

            print("Epoch ", i, ", loss", t_loss)

            if i % print_cycle==0:
                v_loss, v_dict= Model_Func.validation(self, val_dataloader, loss_func, device)
                # pickle.dump(v_dict, open(path+"/v_dict-"+str(self.epoch)+".pkl","wb"))

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

class MultivariateSimulationStudyBaseline_Model(torch.nn.Module):

    def __init__(self, device:torch.device, input_dim, classes=3, lr=0.001):
        super().__init__()

        assert isinstance(input_dim, tuple) or isinstance(input_dim, List), "input_dim must be a list or tuple"
        assert len(input_dim)>1 and len(input_dim)<4, "input_dim must have at least 2 dimensions"
        for i in input_dim:
            assert isinstance(i, int), "input_dim must contain int"

        assert isinstance(device, torch.device), "device is not a torch.device"

        self.device=device
        if len(input_dim)==2:
            self.layer_0= torch.nn.Linear(functools.reduce(lambda a,b: a*b, input_dim), classes)
        else:
            self.layer_0= torch.nn.Linear(input_dim[1]*input_dim[2], classes)

        self.epoch=1
        self.v_dicts=[]
        self.t_dicts=[]
        self.optimiser= None
        self.lr= lr

    def forward(self,x):
        x= x.view(x.shape[0], -1)
        return self.layer_0(x).softmax(dim=1)

    def prediction_procedure(self, dataloader, dict_flag=False):

        device= self.device

        return Model_Func.prediction(self, dataloader, device, dict_flag)

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

            print("Epoch ", i, ", loss", t_loss)

            if i % print_cycle==0:
                v_loss, v_dict= Model_Func.validation(self, val_dataloader, loss_func, device)
                # pickle.dump(v_dict, open(path+"/v_dict-"+str(self.epoch)+".pkl","wb"))

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
