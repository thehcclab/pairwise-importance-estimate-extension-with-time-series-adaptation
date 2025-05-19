import numpy as np
from sklearn.metrics import classification_report
import torcheeg.models
import torch
import os
from tqdm import tqdm
import Models.model_func as Model_Func
from typing import List
from sklearn.preprocessing import MinMaxScaler, RobustScaler

def calculate_accuracy(ys, predictions):
    
    c1_indices= np.nonzero(ys)[0]
    c0_indices= np.nonzero(ys==0)[0]

    c0_count=len(np.nonzero(predictions[c0_indices]==0)[0])
    c1_count=len(np.nonzero(predictions[c1_indices]==1)[0])
    print(c0_count, c1_count)
    print()

    c0_acc= c0_count / len(c0_indices)
    c1_acc= c1_count / len(c1_indices)

    balanced_acc= (c0_acc+c1_acc) /2

    return c0_acc, c1_acc, balanced_acc

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

def Grad_AUC_with_multivar_grad(grads, epoch, X, x_shape=1,scaling=False):

    channels=np.arange(X.shape[x_shape+1])

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


    return dictionary

def Grad_ROC_with_multivar_grad(grads, epoch, X, x_shape=1, scaling=False):

    channels=np.arange(X.shape[x_shape+1])

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
            dictionary[key]= scaler.transform(dictionary[key].reshape(-1,1)).reshape(-1)
    else:
        epoch= return_epoch_stat(grad_all, "sum")
        dictionary={}
        for idx, name in zip(range(len(channels)), channels):
            rolled= np.roll( epoch[:,:, idx], 1 )
            rolled[0]=0.
            diff= epoch[:,:, idx]-rolled
            dictionary[channels[idx]]= np.trapz(abs(diff), axis=0)


    return dictionary

def Grad_STD_with_multivar_grad(grads, epoch, X, x_shape=1,scaling=False):

    channels=np.arange(X.shape[x_shape+1])

    grad_all= np.array(grads).reshape(epoch,-1,X.shape[x_shape], X.shape[x_shape+1])
    
    if scaling:
        scaler= MinMaxScaler()
    
        epoch= return_epoch_stat(grad_all, "std")
        dictionary={}
        for idx, name in zip(range(len(channels)), channels):
            dictionary[name]= np.trapz(abs(epoch[:, :, idx]), axis=0)
            scaler.partial_fit(dictionary[name].reshape(-1,1))


        for key in dictionary.keys():
            dictionary[key]= scaler.transform(dictionary[key].reshape(-1,1)).reshape(-1)
    else:
        epoch= return_epoch_stat(grad_all, "std")
        dictionary={}
        for idx, name in zip(range(len(channels)), channels):
            dictionary[name]= np.trapz(abs(epoch[:, :, idx]), axis=0)


    return dictionary

def eeg_DF_train(model, dataloader, loss_func, device):
    model.train()

    losses= []
    outputs= []
    y_real= []

#     t=tqdm(dataloader, desc="Batch")
    t=dataloader

    for x, y in t:
        model.optimiser.zero_grad()

        x= x.to(device, dtype=torch.float)
        x= x.unsqueeze(1)
#         print("train", x)
        y= y.to(device, dtype=torch.long)

        output= model(x)
        loss= loss_func(output, y)
        l1_loss= model.l1_lambda * model.compute_l1_loss() # concatenated tensors
        l2_loss= model.l2_lambda * model.compute_l2_loss()

        loss += l1_loss
        loss += l2_loss

        loss.backward()
        model.optimiser.step()

        losses += [loss.item()]
        outputs += [output.argmax(dim=1).cpu().detach().numpy()]
        y_real += [y.cpu().detach().numpy()]
    if len(losses)==1:
        return losses[0], outputs[0], y_real[0]
    else:
        return np.array(losses).mean(), np.concatenate(outputs), np.concatenate(y_real)

def eeg_train(model, dataloader, loss_func, device):
    model.train()

    losses= []
    outputs= []
    y_real= []

#     t=tqdm(dataloader, desc="Batch")
    t=dataloader

    for x, y in t:
        model.optimiser.zero_grad()

        x= x.to(device, dtype=torch.float)
        x= x.unsqueeze(1)
#         print("train", x)
        y= y.to(device, dtype=torch.long)

        output= model(x)
        loss= loss_func(output, y)

        loss.backward()
        model.optimiser.step()

        losses += [loss.item()]
        outputs += [output.argmax(dim=1).cpu().detach().numpy()]
        y_real += [y.cpu().detach().numpy()]
    if len(losses)==1:
        return losses[0], outputs[0], y_real[0]
    else:
        return np.array(losses).mean(), np.concatenate(outputs), np.concatenate(y_real)

def eeg_grad_train(model, dataloader, loss_func, device):
    model.train()

    losses= []
    outputs= []
    y_real= []

#     t=tqdm(dataloader, desc="Batch")
    t=dataloader

    for x, y in t:
        model.optimiser.zero_grad()

        x= x.to(device, dtype=torch.float)
        x= x.unsqueeze(1)
#         print("train", x)
        y= y.to(device, dtype=torch.long)

        output= model(x)
        loss= loss_func(output, y)

        loss.backward()
        model.store_IE_gradient()
        model.optimiser.step()

        losses += [loss.item()]
        outputs += [output.argmax(dim=1).cpu().detach().numpy()]
        y_real += [y.cpu().detach().numpy()]
    if len(losses)==1:
        return losses[0], outputs[0], y_real[0]
    else:
        return np.array(losses).mean(), np.concatenate(outputs), np.concatenate(y_real)
    
def eeg_val(model, val_loader, loss_func, device):
    model.eval()
#     loss_avg = 0
#     counter = 0
    y_real= []
    outputs= []
    losses= []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device, dtype=torch.float)
            x = x.unsqueeze(1)
#             print("val", x)
            y = y.to(device, dtype=torch.long)

            output = model(x)
            loss= loss_func(output, y)

            losses += [loss.item()]
            outputs += [output.argmax(dim=1).cpu().detach().numpy()]
            y_real += [y.cpu().detach().numpy()]
    if len(losses)==1:
        return losses[0], outputs[0], y_real[0]
    else:
        return np.array(losses).mean(), np.concatenate(outputs), np.concatenate(y_real)
    
def eeg_validation(model, loader, loss_func, device):
    loss, o, y_real= eeg_val(model, loader, loss_func, device)
    dictionary= classification_report(y_real, o, output_dict=True, zero_division=0)

    return loss, dictionary

def eeg_prediction(model, loader, device, dict_flag=False):
    model.eval()
    predictions= []
    y_real= []

    with torch.no_grad():
        for x, y in loader:
#             print(x)
            x= x.to(device, dtype=torch.float)
            x= x.unsqueeze(1)
#             x= x.to(device, dytpe=torch.float, non_blocking=False, copy=False, memory_format=torch.preserve_format)

            output= model(x)

#             print(output.shape)

            predictions += [output.argmax(dim=1).cpu().detach().numpy()]
            y_real += [y.cpu().detach().numpy()]

    predictions= np.concatenate(predictions)
    y_real= np.concatenate(y_real)

    if dict_flag:
        dictionary= classification_report(y_real, predictions, output_dict=True, zero_division=0)
        return predictions, dictionary
    else:
        classification_report(y_real, predictions, zero_division=0)
        return predictions
    
class EEGNet_Wrapper(torch.nn.Module):
    def __init__(self, device: torch.device, eegnet: torcheeg.models.EEGNet):

        super().__init__()

        assert isinstance(device, torch.device)
        assert isinstance(eegnet, torch.nn.Module) # torcheeg.models.EEGNet
        self.device= device
        self.eegnet= eegnet
        self.t_dicts=[]
        self.v_dicts=[]
        self.epoch=1
        self.optimiser=None
        
    def forward(self, x):
        return self.eegnet(x)

    def set_optimiser(self, optimiser):

        self.optimiser= optimiser

    def prediction_procedure(self, dataloader, dict_flag=False):

        device= self.device

        return eeg_prediction(self, dataloader, device, dict_flag)

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
                v_loss, v_dict= eeg_validation(self, val_dataloader, loss_func, device)
                # pickle.dump(v_dict, open(path+"/v_dict-"+str(self.epoch)+".pkl","wb"))

                self.v_dicts += [v_dict]

                t_acc, v_acc= t_dict["accuracy"], v_dict["accuracy"]
                t_recall, v_recall= t_dict["macro avg"]["recall"], v_dict["macro avg"]["recall"]
                t_prec, v_prec= t_dict["macro avg"]["precision"], v_dict["macro avg"]["precision"]
                t_f1, v_f1= t_dict["macro avg"]["f1-score"], v_dict["macro avg"]["f1-score"]

                print("Epoch: ", i)
                print("t_loss: ", t_loss,", v_loss: ", v_loss)
                print("t_acc: ", t_acc,", v_acc: ", v_acc)
                print("t_recall: ", t_recall,", v_recall: ", v_recall)
                print("t_prec: ", t_prec, ", v_prec: ", v_prec)
                print("t_f: ", t_f1,", v_f: ", v_f1)
                print("////////")

            self.epoch += 1
            
class EEGNet_IE_HP_Wrapper(EEGNet_Wrapper):
    """Simultaneous method"""
    def __init__(self, device:torch.device, eegnet:torcheeg.models.EEGNet, input_dim:List[int]):
        assert isinstance(device, torch.device), "device is not a torch.device"
        assert isinstance(eegnet, torcheeg.models.EEGNet), "eegnet is not a torcheeg.models.EEGNet"
        super().__init__(device, eegnet)

        assert len(input_dim)==2, "Expecting a list with length 2"
        for i in input_dim:
            assert isinstance(i, int), "Expected i to be an int"

        self.IE_weights= torch.empty(input_dim, dtype=torch.float32, requires_grad=True, device=device)
        self.IE_weights= torch.nn.Parameter(torch.nn.init.ones_(self.IE_weights))
        self.IE_grad= []

    def set_IE_weights(self, ie_w):
        self.IE_weights= torch.nn.Parameter(ie_w)

    def return_IE_weights(self):

        return self.IE_weights.detach().cpu().numpy()
    
    def return_IE_gradient(self):
        return self.IE_weights.grad

    def return_IE_grad(self):
        return self.IE_grad
    
    def store_IE_gradient(self):
        if self.return_IE_gradient()!=None:
            self.IE_grad.append( self.return_IE_gradient().detach().cpu().numpy()  )

    def IE_grad_setting(self, boolean):
        self.IE_weights.requires_grad_(boolean)
        self.IE_weights.grad= None

    def forward(self, x):
#         print("IE weights", self.return_layer_weights())
        new_input= x * self.IE_weights

        return self.eegnet(new_input)


    
class EEGNet_IE_EEG_Wrapper(EEGNet_Wrapper):
    """Simultaneous method"""
    def __init__(self, device:torch.device, eegnet:torcheeg.models.EEGNet, input_dim:int):
        assert isinstance(device, torch.device), "device is not a torch.device"
        assert isinstance(eegnet, torcheeg.models.EEGNet), "eegnet is not a torcheeg.models.EEGNet"
        super().__init__(device, eegnet)

        assert isinstance(input_dim, int), "input_dim is not an int"

        self.IE_weights= torch.empty(input_dim, dtype=torch.float32, requires_grad=True, device=device)
        self.IE_weights= torch.nn.Parameter(torch.nn.init.ones_(self.IE_weights))

    def return_IE_weights(self):

        return self.IE_weights.detach().cpu().numpy()

    def forward(self, x):
#         print("IE weights", self.return_layer_weights())
        new_input= x * self.IE_weights.view(1,-1,1)

        return self.eegnet(new_input)
    
class EEGNet_IE_TS_Wrapper(EEGNet_Wrapper):
    """Simultaneous method"""
    def __init__(self, device:torch.device, eegnet:torcheeg.models.EEGNet, input_dim:int):
        assert isinstance(device, torch.device), "device is not a torch.device"
        assert isinstance(eegnet, torcheeg.models.EEGNet), "eegnet is not a torcheeg.models.EEGNet"
        super().__init__(device, eegnet)

        assert isinstance(input_dim, int), "input_dim is not an int"

        self.IE_weights= torch.empty(input_dim, dtype=torch.float32, requires_grad=True, device=device)
        self.IE_weights= torch.nn.Parameter(torch.nn.init.ones_(self.IE_weights))

    def return_IE_weights(self):

        return self.IE_weights.detach().cpu().numpy()

    def forward(self, x):
#         print("IE weights", self.return_layer_weights())
        new_input= x * self.IE_weights

        return self.eegnet(new_input)
    
class EEGNet_DF_HP_Wrapper(EEGNet_Wrapper):
    def __init__(self, device: torch.device, eegnet: torcheeg.models.EEGNet,
                 input_dim: List[int], l1_lambda=1.0, l2_lambda=0.1):

        assert isinstance(device, torch.device), "device is not a torch.device"
        assert isinstance(eegnet, torcheeg.models.EEGNet), "eegnet is not a torcheeg.models.EEGNet"
        super().__init__(device, eegnet)
        
        assert isinstance(l1_lambda, float), "Expecting float for l1_lambda"
        assert isinstance(l2_lambda, float), "Expecting float for l2_lambda"
        assert len(input_dim)==2, "Only accept multivariate data of 2 dimensions"

        for i in input_dim:
            assert isinstance(i, int), f"{i} is not an int"
        
        self.l1_lambda= 1.0
        self.l2_lambda= 0.1
        
        self.DF_weights= torch.empty(input_dim, dtype=torch.float32, requires_grad=True)
        self.DF_weights= torch.nn.Parameter ( torch.nn.init.uniform_(self.DF_weights) )

    def return_DF_weights(self):
        return self.DF_weights.detach().cpu().numpy()

    def compute_l1_loss(self):
        return torch.abs(self.DF_weights).sum()

    def compute_l2_loss(self):
        return (self.DF_weights*self.DF_weights).sum()

    def forward(self,x):
        new_input= x * self.DF_weights
        output= self.eegnet(new_input)
        return output

class EEGNet_DF_EEG_Wrapper(EEGNet_Wrapper):
    def __init__(self, device: torch.device, eegnet: torcheeg.models.EEGNet,
                 input_dim: int, l1_lambda=1.0, l2_lambda=0.1):

        assert isinstance(device, torch.device), "device is not a torch.device"
        assert isinstance(eegnet, torcheeg.models.EEGNet), "eegnet is not a torcheeg.models.EEGNet"
        super().__init__(device, eegnet)
        
        assert isinstance(l1_lambda, float), "Expecting float for l1_lambda"
        assert isinstance(l2_lambda, float), "Expecting float for l2_lambda"
        assert isinstance(input_dim, int), "input_dim is not an int"
        
        self.l1_lambda= 1.0
        self.l2_lambda= 0.1
        
        self.DF_weights= torch.empty(input_dim, dtype=torch.float32, requires_grad=True)
        self.DF_weights= torch.nn.Parameter ( torch.nn.init.uniform_(self.DF_weights) )

    def return_DF_weights(self):
        return self.DF_weights.detach().cpu().numpy()

    def compute_l1_loss(self):
        return torch.abs(self.DF_weights).sum()

    def compute_l2_loss(self):
        return (self.DF_weights*self.DF_weights).sum()

    def forward(self,x):
        new_input= x * self.DF_weights.view(1,-1,1)
        return self.eegnet(new_input)

class EEGNet_DF_TS_Wrapper(EEGNet_Wrapper):
    def __init__(self, device: torch.device, eegnet: torcheeg.models.EEGNet,
                 input_dim: int, l1_lambda=1.0, l2_lambda=0.1):

        assert isinstance(device, torch.device), "device is not a torch.device"
        assert isinstance(eegnet, torcheeg.models.EEGNet), "eegnet is not a torcheeg.models.EEGNet"
        super().__init__(device, eegnet)
        
        assert isinstance(l1_lambda, float), "Expecting float for l1_lambda"
        assert isinstance(l2_lambda, float), "Expecting float for l2_lambda"
        assert isinstance(input_dim, int), "input_dim is not an int"
        
        self.l1_lambda= 1.0
        self.l2_lambda= 0.1
        
        self.DF_weights= torch.empty(input_dim, dtype=torch.float32, requires_grad=True)
        self.DF_weights= torch.nn.Parameter ( torch.nn.init.uniform_(self.DF_weights) )

    def return_DF_weights(self):
        return self.DF_weights.detach().cpu().numpy()

    def compute_l1_loss(self):
        return torch.abs(self.DF_weights).sum()

    def compute_l2_loss(self):
        return (self.DF_weights*self.DF_weights).sum()

    def forward(self,x):
        new_input= x * self.DF_weights
        return self.eegnet(new_input)
    
class EEGNet_NeuralFS_HP_Wrapper(EEGNet_Wrapper):
    def __init__(self, device: torch.device, eegnet: torcheeg.models.EEGNet, input_dim: List[int], nonlinear_func: torch.nn.Module):
        
        assert isinstance(device, torch.device), "device is nto a torch.device"
        assert isinstance(eegnet, torcheeg.models.EEGNet), "device is nto a torch.device"
        super().__init__(device, eegnet)

        assert isinstance(nonlinear_func, torch.nn.Module), "Expecting torch.nn.Module for the model parameter"
        assert len(input_dim)==2, "Only accept multivariate data of 2 dimensions"

        for i in input_dim:
            assert isinstance(i, int), f"{i} is not an int"

        self.nonlinear_func= nonlinear_func

        self.pairwise_weights= torch.empty(input_dim, dtype=torch.float32, requires_grad=True)
        self.pairwise_weights=torch.nn.Parameter( torch.nn.init.uniform_(self.pairwise_weights) )

    def return_pairwise_weights(self):
        return self.pairwise_weights.detach().cpu().numpy()

    def Thresholded_Linear(self, x, threshold=0.2):

        return ( (x > threshold) | (x < -threshold) ) * x

    def forward(self,x):
#         print("original x", x.shape)
        x= x.squeeze(1)
#         print("x", x.shape)
        nonlinear_output= self.nonlinear_func(x)

#         print("nonlinear_output", nonlinear_output.shape)
#         print("pairwise_weights", self.pairwise_weights.shape)

        pairwise_connected_output= nonlinear_output * self.pairwise_weights

#         print("pariwise_connected_output", pairwise_connected_output.shape)
        
        thresholded_pairwise_output= self.Thresholded_Linear(pairwise_connected_output)

        selected_input= x * thresholded_pairwise_output
        
#         print("selected input",selected_input.shape)

        output= self.eegnet(selected_input.squeeze(0).unsqueeze(1))

        return output
    
class EEGNet_NeuralFS_EEG_Wrapper(EEGNet_Wrapper):
    def __init__(self, device: torch.device, eegnet: torcheeg.models.EEGNet, input_dim: List[int], nonlinear_func: torch.nn.Module):
        
        assert isinstance(device, torch.device), "device is nto a torch.device"
        assert isinstance(eegnet, torcheeg.models.EEGNet), "device is nto a torch.device"
        super().__init__(device, eegnet)

        assert isinstance(nonlinear_func, torch.nn.Module), "Expecting torch.nn.Module for the model parameter"
        assert isinstance(input_dim, int), "input_dim is not an int"

        self.nonlinear_func= nonlinear_func

        self.pairwise_weights= torch.empty(input_dim, dtype=torch.float32, requires_grad=True)
        self.pairwise_weights=torch.nn.Parameter( torch.nn.init.uniform_(self.pairwise_weights) )

    def return_pairwise_weights(self):
        return self.pairwise_weights.detach().cpu().numpy()

    def Thresholded_Linear(self, x, threshold=0.2):

        return ( (x > threshold) | (x < -threshold) ) * x

    def forward(self,x):
#         print("original x", x.shape)
        x= x.squeeze(1)
#         print("x", x.shape)
        nonlinear_output= self.nonlinear_func(x)

#         print("nonlinear_output", nonlinear_output.shape)
#         print("pairwise_weights", self.pairwise_weights.shape)

        pairwise_connected_output= nonlinear_output * self.pairwise_weights.view(1,-1,1)

#         print("pariwise_connected_output", pairwise_connected_output.shape)
        
        thresholded_pairwise_output= self.Thresholded_Linear(pairwise_connected_output)

        selected_input= x * thresholded_pairwise_output
        
#         print("selected input",selected_input.shape)

        output= self.eegnet(selected_input.squeeze(0).unsqueeze(1))

        return output
    
class EEGNet_NeuralFS_TS_Wrapper(EEGNet_Wrapper):
    def __init__(self, device: torch.device, eegnet: torcheeg.models.EEGNet, input_dim: List[int], nonlinear_func: torch.nn.Module):
        
        assert isinstance(device, torch.device), "device is nto a torch.device"
        assert isinstance(eegnet, torcheeg.models.EEGNet), "device is nto a torch.device"
        super().__init__(device, eegnet)

        assert isinstance(nonlinear_func, torch.nn.Module), "Expecting torch.nn.Module for the model parameter"
        assert isinstance(input_dim, int), "input_dim is not an int"

        self.nonlinear_func= nonlinear_func

        self.pairwise_weights= torch.empty(input_dim, dtype=torch.float32, requires_grad=True)
        self.pairwise_weights=torch.nn.Parameter( torch.nn.init.uniform_(self.pairwise_weights) )

    def return_pairwise_weights(self):
        return self.pairwise_weights.detach().cpu().numpy()

    def Thresholded_Linear(self, x, threshold=0.2):

        return ( (x > threshold) | (x < -threshold) ) * x

    def forward(self,x):
#         print("original x", x.shape)
        x= x.squeeze(1)
#         print("x", x.shape)
        nonlinear_output= self.nonlinear_func(x)

#         print("nonlinear_output", nonlinear_output.shape)
#         print("pairwise_weights", self.pairwise_weights.shape)

        pairwise_connected_output= nonlinear_output * self.pairwise_weights

#         print("pariwise_connected_output", pairwise_connected_output.shape)
        
        thresholded_pairwise_output= self.Thresholded_Linear(pairwise_connected_output)

        selected_input= x * thresholded_pairwise_output
        
#         print("selected input",selected_input.shape)

        output= self.eegnet(selected_input.squeeze(0).unsqueeze(1))

        return output