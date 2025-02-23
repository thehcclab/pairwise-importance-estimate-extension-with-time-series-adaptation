import torch
import logging
import os
import Models.model_func as Model_Func
import pickle
from tqdm import tqdm
import numpy as np

from sklearn.metrics import classification_report

# def DF_train_rnn(model, dataloader, optimiser, loss_func, device):
#     model.train()
#
#     losses= []
#     outputs= []
#     y_real= []
#
#     # t=tqdm(dataloader, desc="Batch")
#     t=dataloader
#
#     for x, y in t:
#         optimiser.zero_grad()
#
#         x= x.to(device, dtype=torch.float)
# #         print("train", x)
#         y= y.to(device, dtype=torch.long)
#
#         y= y.repeat(x.shape[1])
#
#         output= model(x)
#         # print("output", output.shape, ", y", y.shape)
#         loss= loss_func(output, y)
#
#         l1_loss= model.l1_lambda * model.compute_l1_loss() # concatenated tensors
#
#         l2_loss= model.l2_lambda * model.compute_l2_loss()
#
#         loss += l1_loss
#         loss += l2_loss
#
#         loss.backward()
#         optimiser.step()
#
#         losses += [loss.item()]
#         outputs += [output.argmax(dim=1).cpu().detach().numpy()]
#         y_real += [y.cpu().detach().numpy()]
#     if len(losses)==1:
#         return losses[0], outputs[0], y_real[0]
#     else:
#         return np.array(losses).mean(), np.concatenate(outputs), np.concatenate(y_real)
#
#
#
#
# def train_rnn(model, dataloader, optimiser, loss_func, device):
#     model.train()
#
#     losses= []
#     outputs= []
#     y_real= []
#
#     # t=tqdm(dataloader, desc="Batch")
#     t=dataloader
#
#     for x, y in t:
#         optimiser.zero_grad()
#
#         x= x.to(device, dtype=torch.float)
# #         print("train", x)
#         y= y.to(device, dtype=torch.long)
#
#         y= y.repeat(x.shape[1])
#
#         output= model(x)
#         # print("output", output.shape, ", y", y.shape)
#         loss= loss_func(output, y)
#
#         loss.backward()
#         optimiser.step()
#
#         losses += [loss.item()]
#         outputs += [output.argmax(dim=1).cpu().detach().numpy()]
#         y_real += [y.cpu().detach().numpy()]
#     if len(losses)==1:
#         return losses[0], outputs[0], y_real[0]
#     else:
#         return np.array(losses).mean(), np.concatenate(outputs), np.concatenate(y_real)
#
# def val_rnn(model, val_loader, loss_func, device):
#     model.eval()
# #     loss_avg = 0
# #     counter = 0
#     y_real= []
#     outputs= []
#     losses= []
#     with torch.no_grad():
#         for x, y in val_loader:
#             x = x.to(device, dtype=torch.float)
# #             print("val", x)
#             y = y.to(device, dtype=torch.long)
#
#             y= y.repeat(x.shape[1])
#
#             output = model(x)
#             loss= loss_func(output, y)
#
#             losses += [loss.item()]
#             outputs += [output.argmax(dim=1).cpu().detach().numpy()]
#             y_real += [y.cpu().detach().numpy()]
#     if len(losses)==1:
#         return losses[0], outputs[0], y_real[0]
#     else:
#         return np.array(losses).mean(), np.concatenate(outputs), np.concatenate(y_real)
#
# def prediction(model, loader, device, print_flag=False):
#     model.eval()
#     predictions= []
#     y_real= []
#
#     with torch.no_grad():
#         for x, y in loader:
# #             print(x)
#             x= x.to(device, dtype=torch.float)
# #             x= x.to(device, dytpe=torch.float, non_blocking=False, copy=False, memory_format=torch.preserve_format)
#             y= y.repeat(x.shape[1])
#             # print("y", y.shape)
#
#             output= model(x)
#
# #             print(output.shape)
#
#             predictions += [output.argmax(dim=1).cpu().detach().numpy()]
#             y_real += [y.cpu().detach().numpy()]
#
#     predictions= np.concatenate(predictions)
#     y_real= np.concatenate(y_real)
#
#     # print(f"y_real: {y_real.shape}, pred: {predictions.shape}")
#
#     if print_flag:
#         dictionary= classification_report(y_real, predictions, output_dict=True, zero_division=0)
#         return predictions, dictionary
#     else:
#         classification_report(y_real, predictions, zero_division=0)
#         return predictions

class HHARBaseline_Model(torch.nn.Module):
    def __init__(self, device:torch.device, input_dim:int, classes:int, rnn_hidden=64, lr=0.0001):
        super().__init__()

        assert isinstance(input_dim, int), "input_dim is not an int"
        assert isinstance(device, torch.device), "device is not a torch.device"

        self.device=device
##################### TRIED once

        # self.rnn=torch.nn.GRU(input_dim,rnn_hidden=64,batch_first=True, num_layers=2)
        # self.dropout= torch.nn.Dropout(0.1)
        #
        # self.layer_0= torch.nn.Linear(rnn_hidden, 256) # original
        # self.layer_1= torch.nn.Linear(256, 64)
        # self.output= torch.nn.Linear(64, classes)
        #
        # self.log_soft= torch.nn.LogSoftmax(dim=1)
######################

        self.rnn= torch.nn.LSTM(input_dim, rnn_hidden, batch_first=True)
        self.dropout= torch.nn.Dropout(0.1)
        self.layer_2= torch.nn.Linear(rnn_hidden,classes)
        self.log_soft= torch.nn.LogSoftmax(dim=1)


        self.lr= lr
        self.optimiser= None
        self.epoch=1
        self.v_dicts=[]
        self.t_dicts=[]

    def forward(self,x):
#################### TRIED ONCE
        # print("x", x.shape)
        # print("x unsqueeze", x.unsqueeze(0).shape)
        # rnn_output, states= self.rnn(x)
        #
        # # print("rnn_output", rnn_output.shape)
        # rnn_output= rnn_output.squeeze(0)
        # rnn_output= self.dropout(rnn_output)
        # # output= self.layer_2(self.layer_1(rnn_output)).softmax(dim=1)
        # output= self.layer_1(self.layer_0(rnn_output)).tanh() # original
        # output= self.log_soft(output)

        # print("output softmax", output.shape)
############################

        rnn_output, states= self.rnn(x)
        rnn_output= rnn_output.squeeze(0)
        rnn_output= self.dropout(rnn_output)
        output= self.layer_2(rnn_output)
        output= self.log_soft(output)

        return output

        # return self.layer_2( self.layer_1( self.layer_0(x) ) ).softmax(dim=1)
        # return self.layer_1( self.layer_0(x) ).softmax(dim=1)

    def set_optimiser(self, optimiser):
        self.optimiser= optimiser

    def prediction_procedure(self, dataloader, dict_flag=False):

        device= self.device

        # return prediction(self, dataloader, device, print_flag)
        return Model_Func.prediction_rnn(self, dataloader, device, dict_flag)

    def training_procedure(self, iteration, train_dataloader, val_dataloader, path, loss_func, optimiser, train_func, print_cycle):
        device= self.device

        # path= path+"-dictionary"
        # if not os.path.exists(path):
        #     os.makedirs(path)

        if self.optimiser==None:
            self.optimiser= optimiser

        for i in tqdm(range(iteration), desc="Iterations"):
            t_loss, t_dict= Model_Func.training(self, train_dataloader, train_func, loss_func, device)

            self.t_dicts += [t_dict]
            # pickle.dump(t_dict, open(path+"/t_dict-"+str(self.epoch)+".pkl", "wb"))

            print("Epoch ", self.epoch, ", loss", t_loss)

            if self.epoch % print_cycle==0:
                # v_loss, predictions, y_real= val_rnn(self, val_dataloader, loss_func, device)
                # v_dict= classification_report(y_real, predictions, output_dict=True, zero_division=0)
                v_loss, v_dict= Model_Func.validation_rnn(self, val_dataloader, loss_func, device)

                # pickle.dump(v_dict, open(path+"/v_dict-"+str(self.epoch)+".pkl","wb"))
                self.v_dicts += [v_dict]

                t_acc, v_acc= t_dict["accuracy"], v_dict["accuracy"]
                t_recall, v_recall= t_dict["macro avg"]["recall"], v_dict["macro avg"]["recall"]
                t_prec, v_prec= t_dict["macro avg"]["precision"], v_dict["macro avg"]["precision"]
                t_f1, v_f1= t_dict["macro avg"]["f1-score"], v_dict["macro avg"]["f1-score"]

                # print("Epoch: ", i)
                # print("t_loss: ", t_loss,", v_loss: ", v_loss)
                # print("t_acc: ", t_acc,", v_acc: ", v_acc)
                # print("t_recall: ", t_recall,", v_recall: ", v_recall)
                # print("t_prec: ", t_prec, ", v_prec: ", v_prec)
                # print("t_f: ", t_f1,", v_f: ", v_f1)
                # print("////////")

                logging.info(f"Epoch: {i}")
                logging.info(f"t_loss: {t_loss}, v_loss: {v_loss}")
                logging.info(f"t_acc: {t_acc}, v_acc: {v_acc}")
                logging.info(f"t_recall: {t_recall}, v_recall: {v_recall}")
                logging.info(f"t_prec: {t_prec}, v_prec: {v_prec}")
                logging.info(f"t_f: {t_f1}, v_f: {v_f1}")
                logging.info("//////////")

            self.epoch += 1
