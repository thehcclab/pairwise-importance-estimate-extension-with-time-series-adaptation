import torch
from typing import List
import logging
import os
import Models.model_func as Model_Func
import pickle
from tqdm import tqdm
import yaml

class Multivariate_Model(torch.nn.Module):
    def __init__(self, device: torch.device):

        super().__init__()

        assert isinstance(device, torch.device)

        self.device= device

        self.t_dicts=[]
        self.v_dicts=[]
        self.epoch=1
        self.optimiser=None

    def set_optimiser(self, optimiser):

        self.optimiser= optimiser

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

            print("Epoch ", i, ", loss", t_loss)

            if i % print_cycle==0:
                v_loss, v_dict= Model_Func.validation_rnn(self, val_dataloader, loss_func, device)
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



class MultivariateSeries_DF(Multivariate_Model):
    def __init__(self, device: torch.device, input_dim: int, model: torch.nn.Module, l1_lambda=1.0, l2_lambda=0.1):
        """
        input_dim: dimensions of input data, usually in the form of [time_domain_dim, feature_domain_dim]
        model: Existing Pytorch torch.nn.Module model
        l1_lambda, l2_lambda: variables for the parameter of the Elastic Net regularisation
        """
        assert isinstance(device, torch.device), "device is nto a torch.device"
        super().__init__(device)

        # assert isinstance(device, torch.device)
        assert isinstance(model, torch.nn.Module), "Expecting torch.nn.Module for the model parameter"
        assert isinstance(l1_lambda, float), "Expecting float for l1_lambda"
        assert isinstance(l2_lambda, float), "Expecting float for l2_lambda"
        assert isinstance(input_dim, int), "Expecting int for input_dim"
#         assert len(input_dim)==2, "Only accept multivariate data of 2 dimensions"

#         for i in input_dim:
#             assert isinstance(i, int), f"{i} is not an int"

        # self.device=device
        self.model= model

        self.l1_lambda= l1_lambda
        self.l2_lambda= l2_lambda

# Creating a non-linear model which expects input of size, [batch, timesteps, features]

        self.DF_weights= torch.empty(input_dim, dtype=torch.float32, requires_grad=True)
        self.DF_weights= torch.nn.Parameter ( torch.nn.init.uniform_(self.DF_weights) )

#                     self.mode=mode #"ts" or "dc"
        logging.info("Nonlinear Model")

    def return_DF_weights(self):
        return self.DF_weights.detach().cpu().numpy()

    def compute_l1_loss(self):
        return torch.abs(self.DF_weights).sum()

    def compute_l2_loss(self):
        return (self.DF_weights*self.DF_weights).sum()

    def forward(self,x):
#         new_input= torch.matmul(self.DF_ts, x)
#         print(self.DF_ts_weights.view(1,-1,1).shape)
#         new_input= self.DF_ts_weights.view(1,-1,1) * x
#         new_input= new_input * self.DF_dc_weights
        new_input= x * self.DF_weights
        output= self.model(new_input)
        return output

class MultivariateSeries_NeuralFS(Multivariate_Model):
    def __init__(self, device: torch.device, input_dim: List[int], decision_net: torch.nn.Module, nonlinear_func: torch.nn.Module):
        assert isinstance(device, torch.device), "device is nto a torch.device"
        super().__init__(device)

        assert isinstance(nonlinear_func, torch.nn.Module), "Expecting torch.nn.Module for the model parameter"
        assert isinstance(decision_net, torch.nn.Module), "Expecting torch.nn.Module for the decision_net parameter"
        assert isinstance(input_dim, int), "Expecting int for input_dim"
#         assert isinstance(mode, str), "Expecting mode to be 'ts', 'dc', 'all' or 'both'"
#         assert len(input_dim)==2, "Only accept multivariate data of 2 dimensions"

#         for i in input_dim:
#             assert isinstance(i, int), f"{i} is not an int"

        self.decision_net= decision_net
        self.nonlinear_func= nonlinear_func

        self.pairwise_weights= torch.empty(input_dim, dtype=torch.float32, requires_grad=True)
        self.pairwise_weights=torch.nn.Parameter( torch.nn.init.uniform_(self.pairwise_weights) )

    def return_pairwise_weights(self):
        return self.pairwise_weights.detach().cpu().numpy()

    def Thresholded_Linear(self, x, threshold=0.2):

        return ( (x > threshold) | (x < -threshold) ) * x

    def forward(self,x):
        decision_net_output= self.nonlinear_func(x)

#         print("decision_net_output", decision_net_output.shape)
#         print("pairwise_weights", self.pairwise_weights.shape)

        pairwise_connected_output= decision_net_output * self.pairwise_weights

        thresholded_pairwise_output= self.Thresholded_Linear(pairwise_connected_output)

        selected_input= x * thresholded_pairwise_output

        output= self.decision_net(selected_input)

        return output

class MultivariateSeries_IELayer(Multivariate_Model):
    """Simultaneous method"""
    def __init__(self, device:torch.device, input_dim:int, model:torch.nn.Module):
        assert isinstance(device, torch.device), "device is nto a torch.device"
        super().__init__(device)

        assert isinstance(input_dim, int), "input_dim is not an int"
        assert isinstance(model, torch.nn.Module), "model is not torch.nn.Module"

        self.model= model

        self.IE_weights= torch.empty(input_dim, dtype=torch.float32, requires_grad=True, device=device)
        self.IE_weights= torch.nn.Parameter(torch.nn.init.ones_(self.IE_weights))

    def return_IE_weights(self):

        return self.IE_weights.detach().cpu().numpy()

    def forward(self, x):
#         print("IE weights", self.return_layer_weights())
        new_input= x * self.IE_weights

        return self.model(new_input)

class MultivariateSeries_IEGradient(MultivariateSeries_IELayer):
    def __init__(self, device:torch.device, input_dim:int, model:torch.nn.Module):
        assert isinstance(device, torch.device), "device is nto a torch.device"
        assert isinstance(input_dim, int), "input_dim is not an int"
        assert isinstance(model, torch.nn.Module), "model is not torch.nn.Module"
        super().__init__(device, input_dim, model)

#         assert len(input_dim)==2, "Expecting a list with length 2"
#         for i in input_dim:
#             assert isinstance(i, int), "Expected i to be an int"

        self.IE_grad= []

    def return_IE_gradient(self):
        return self.IE_weights.grad

    def store_IE_gradient(self):
        if self.return_IE_gradient()!=None:
            self.IE_grad.append( self.return_IE_gradient().clone().detach().cpu().numpy()  )

    def IE_grad_setting(self, boolean):
        self.IE_weights.requires_grad_(boolean)
        self.IE_weights.grad= None

    def return_IE_grad(self):
        return self.IE_grad

    def forward(self, x):
        new_input= x * self.IE_weights
        return self.model(new_input)




class Multivariate_DF(Multivariate_Model):
    def __init__(self, device: torch.device, input_dim: List[int], model: torch.nn.Module, l1_lambda=1.0, l2_lambda=0.1, mode="dc"):
        """
        input_dim: dimensions of input data, usually in the form of [time_domain_dim, feature_domain_dim]
        model: Existing Pytorch torch.nn.Module model
        l1_lambda, l2_lambda: variables for the parameter of the Elastic Net regularisation
        mode: 'dc', data channels, or 'ts', time steps.
            'dc'-> measures importance of feature_domain data channels
            'ts'-> measures importance of time_domain channels
        """
        assert isinstance(device, torch.device), "device is nto a torch.device"
        super().__init__(device)

        assert isinstance(model, torch.nn.Module), "Expecting torch.nn.Module for the model parameter"
        assert isinstance(l1_lambda, float), "Expecting float for l1_lambda"
        assert isinstance(l2_lambda, float), "Expecting float for l2_lambda"
        assert isinstance(mode, str), "Expecting mode to be 'ts', 'dc', 'all' or 'both'"
        assert len(input_dim)==2, "Only accept multivariate data of 2 dimensions"

        for i in input_dim:
            assert isinstance(i, int), f"{i} is not an int"

        self.model= model

        self.l1_lambda= l1_lambda
        self.l2_lambda= l2_lambda

# Creating a non-linear model which expects input of size, [batch, timesteps, features]

#         self.DF_dc_weights= torch.empty( input_dim[1], dtype=torch.float32, requires_grad=True  )
#         self.DF_ts_weights= torch.empty( input_dim[0], dtype=torch.float32, requires_grad=True  )


#         if mode=="dc":
# #                         Estimate importance of feature_domain data channels
#             self.DF_dc_weights= torch.nn.Parameter( torch.nn.init.uniform_(self.DF_dc_weights) )
#             self.DF_ts_weights= torch.nn.init.ones_(self.DF_ts_weights)
#             self.DF_weights= self.DF_dc_weights
#         elif mode=="ts":
# #                         Estimate importance of time_domain data channels
#             self.DF_dc_weights= torch.nn.init.ones_(self.DF_dc_weights)
#             self.DF_ts_weights= torch.nn.Parameter( torch.nn.init.uniform_(self.DF_ts_weights) )
#             self.DF_weights= self.DF_ts_weights
#         elif mode=="both":
#             self.DF_ts_weights= torch.nn.Parameter( torch.nn.init.uniform_(self.DF_ts_weights) )
#             self.DF_dc_weights= torch.nn.Parameter( torch.nn.init.uniform_(self.DF_dc_weights) )
#             self.DF_weights= [self.DF_ts_weights, self.DF_dc_weights]
        self.DF_weights= torch.empty(input_dim, dtype=torch.float32, requires_grad=True)
        self.DF_weights= torch.nn.Parameter ( torch.nn.init.uniform_(self.DF_weights) )

#                     self.mode=mode #"ts" or "dc"
        logging.info("Nonlinear Model")

    def return_DF_weights(self):
        return self.DF_weights.detach().cpu().numpy()

    def compute_l1_loss(self):
        return torch.abs(self.DF_weights).sum()

    def compute_l2_loss(self):
        return (self.DF_weights*self.DF_weights).sum()

    def forward(self,x):
#         new_input= torch.matmul(self.DF_ts, x)
#         print(self.DF_ts_weights.view(1,-1,1).shape)
#         new_input= self.DF_ts_weights.view(1,-1,1) * x
#         new_input= new_input * self.DF_dc_weights
        new_input= x * self.DF_weights
        output= self.model(new_input)
        return output

class Multivariate_NeuralFS(Multivariate_Model):
    def __init__(self, device: torch.device, input_dim: List[int], decision_net: torch.nn.Module, nonlinear_func: torch.nn.Module, mode="dc"):
        assert isinstance(device, torch.device), "device is nto a torch.device"
        super().__init__(device)

        assert isinstance(nonlinear_func, torch.nn.Module), "Expecting torch.nn.Module for the model parameter"
        assert isinstance(decision_net, torch.nn.Module), "Expecting torch.nn.Module for the decision_net parameter"
        assert isinstance(mode, str), "Expecting mode to be 'ts', 'dc', 'all' or 'both'"
        assert len(input_dim)==2, "Only accept multivariate data of 2 dimensions"

        for i in input_dim:
            assert isinstance(i, int), f"{i} is not an int"

        self.decision_net= decision_net
        self.nonlinear_func= nonlinear_func

        self.pairwise_weights= torch.empty(input_dim, dtype=torch.float32, requires_grad=True)
        self.pairwise_weights=torch.nn.Parameter( torch.nn.init.uniform_(self.pairwise_weights) )

    def return_pairwise_weights(self):
        return self.pairwise_weights.detach().cpu().numpy()

    def Thresholded_Linear(self, x, threshold=0.2):

        return ( (x > threshold) | (x < -threshold) ) * x

    def forward(self,x):
        decision_net_output= self.nonlinear_func(x)

#         print("decision_net_output", decision_net_output.shape)
#         print("pairwise_weights", self.pairwise_weights.shape)

        pairwise_connected_output= decision_net_output * self.pairwise_weights

        thresholded_pairwise_output= self.Thresholded_Linear(pairwise_connected_output)

        selected_input= x * thresholded_pairwise_output

        output= self.decision_net(selected_input)

        return output

class Multivariate_IELayer(Multivariate_Model):
    """Simultaneous method"""
    def __init__(self, device:torch.device, input_dim:int, model:torch.nn.Module):
        assert isinstance(device, torch.device), "device is nto a torch.device"
        super().__init__(device)

        assert len(input_dim)==2, "Expecting a list with length 2"
        for i in input_dim:
            assert isinstance(i, int), "Expected i to be an int"
        assert isinstance(model, torch.nn.Module), "model is not torch.nn.Module"

        self.model= model

        self.IE_weights= torch.empty(input_dim, dtype=torch.float32, requires_grad=True, device=device)
        self.IE_weights= torch.nn.Parameter(torch.nn.init.ones_(self.IE_weights))

    def set_IE_weights(self, ie_w):
        self.IE_weights= torch.nn.Parameter(ie_w)

    def return_IE_weights(self):

        return self.IE_weights.detach().cpu().numpy()

    def forward(self, x):
#         print("IE weights", self.return_layer_weights())
        new_input= x * self.IE_weights

        return self.model(new_input)

class Multivariate_IEGradient(Multivariate_IELayer):
    def __init__(self, device:torch.device, input_dim:List[int], model:torch.nn.Module, save_path="", fold="", fold_name=""):
        assert isinstance(device, torch.device), "device is nto a torch.device"
        assert len(input_dim)==2, "Only accept multivariate data of 2 dimensions"

        for i in input_dim:
            assert isinstance(i, int), f"{i} is not an int"

        assert isinstance(model, torch.nn.Module), "model is not torch.nn.Module"
        super().__init__(device, input_dim, model)

        self.IE_grad= []


        self.save_path=save_path
        self.fold=fold
        self.fold_name= fold_name
        if fold !="":
            if not os.path.exists(f"{save_path}/fold{fold}"):
                 os.makedirs(f"{save_path}/fold{fold}")
            self.path= f"{save_path}/fold{fold}/{fold_name}-multivar-grads-e{self.epoch}"
            print(f"save_path {self.path}")

    def update_path(self):
        self.path= f"{self.save_path}/fold{self.fold}/{self.fold_name}-multivar-grads-e{self.epoch}"

    def return_IE_gradient(self):
        return self.IE_weights.grad

    def store_IE_gradient(self):
        if self.return_IE_gradient()!=None:
            self.IE_grad.append( self.return_IE_gradient().detach().cpu().numpy()  )

    def IE_grad_setting(self, boolean):
        self.IE_weights.requires_grad_(boolean)
        self.IE_weights.grad= None

    def return_IE_grad(self):
        return self.IE_grad

    def forward(self, x):
        new_input= x * self.IE_weights
        return self.model(new_input)
