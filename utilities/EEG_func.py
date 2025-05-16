import numpy as np
from sklearn.metrics import classification_report
import torcheeg.models
import torch
import os
from tqdm import tqdm
import Models.model_func as Model_Func

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
        assert isinstance(eegnet, torcheeg.models.EEGNet)
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
    