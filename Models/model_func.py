# import tensorflow as tf
import numpy as np
from tqdm import tqdm
import sys
import torch
import pickle

from sklearn.metrics import classification_report

def DF_train(model, dataloader, loss_func, device):
    model.train()
#     counter = 0
#     loss_avg = 0
    losses=[]
    outputs=[]
    y_real=[]

    for x, y in tqdm(dataloader, desc="Batch"):
        model.optimiser.zero_grad()

        x = x.to(device, dtype=torch.float)
        y = y.to(device, dtype=torch.long)

#         print("train")
#         print(x)
        # forward pass
        output = model(x)

        loss = loss_func(output, y)

#       elastic net regularisation
#         df_parameter_tensor= model.DF_weights#parameter as tensors

        l1_loss= model.l1_lambda * model.compute_l1_loss() # concatenated tensors

        l2_loss= model.l2_lambda * model.compute_l2_loss()

        loss += l1_loss
        loss += l2_loss


        # backward pass
        loss.backward()
        model.optimiser.step()

#         print(loss.item())
        losses += [loss.item()]
        outputs += [output.argmax(dim=1).cpu().detach().numpy()]
        y_real += [y.cpu().detach().numpy()]

#         print(losses)
    if len(losses)==1:
        return losses[0], outputs[0], y_real[0]
    else:
        return np.array(losses).mean(), np.concatenate(outputs), np.concatenate(y_real)

def DF_train_rnn(model, dataloader, loss_func, device):
    model.train()
#     counter = 0
#     loss_avg = 0
    losses=[]
    outputs=[]
    y_real=[]

    for x, y in tqdm(dataloader, desc="Batch"):
        model.optimiser.zero_grad()

        x = x.to(device, dtype=torch.float)
        y = y.to(device, dtype=torch.long)

        y = y.squeeze(0)
#         print("train")
#         print(x)
        # forward pass
        output = model(x)

        loss = loss_func(output, y)

#       elastic net regularisation
#         df_parameter_tensor= model.DF_weights#parameter as tensors

        l1_loss= model.l1_lambda * model.compute_l1_loss() # concatenated tensors

        l2_loss= model.l2_lambda * model.compute_l2_loss()

        loss += l1_loss
        loss += l2_loss


        # backward pass
        loss.backward()
        model.optimiser.step()

#         print(loss.item())
        losses += [loss.item()]
        outputs += [output.argmax(dim=1).cpu().detach().numpy()]
        y_real += [y.cpu().detach().numpy()]

#         print(losses)
    if len(losses)==1:
        return losses[0], outputs[0], y_real[0]
    else:
        return np.array(losses).mean(), np.concatenate(outputs), np.concatenate(y_real)

def Thresholded_train(model, dataloader, loss_func, device):
    model.train()
#     counter = 0
#     loss_avg = 0
    losses=[]
    outputs=[]
    y_real=[]


    for x, y in tqdm(dataloader, desc="Batch"):
#         optimiser.zero_grad()

        x = x.to(device, dtype=torch.float)
        y = y.to(device, dtype=torch.long)

        # forward pass
        output = model(x)

        loss = loss_func(output, y)


#         print("before:", model.return_layer_weights()[:5])
        model.cache_IE_weights()

        # backward pass
        model.optimiser.zero_grad()
        loss.backward()
        model.optimiser.step()
#         print("after:", model.IE_weights[:5])

        tmp= model.threshold_layer()
        with torch.no_grad():
            model.IE_weights.copy_(tmp)
#         print("after thresholding:", model.IE_weights[:5])
#         print()

#         print(loss.item())
        losses += [loss.item()]
        outputs += [output.argmax(dim=1).cpu().detach().numpy()]
        y_real += [y.cpu().detach().numpy()]





#         print(losses)
    if len(losses)==1:
        return losses[0], outputs[0], y_real[0]
    else:
        return np.array(losses).mean(), np.concatenate(outputs), np.concatenate(y_real)

def grad_train(model, dataloader, loss_func, device):
    model.train()

    losses= []
    outputs= []
    y_real= []

    t=tqdm(dataloader, desc="Batch")
    # t=dataloader

    for x, y in t:
        model.optimiser.zero_grad()

        x= x.to(device, dtype=torch.float)
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

def grad_train_rnn_alt(model, dataloader, loss_func, device):
    model.train()

    losses= []
    outputs= []
    y_real= []

    t=tqdm(dataloader, desc="Batch")
    # t=dataloader

    for x, y in t:
        model.optimiser.zero_grad()

        x= x.to(device, dtype=torch.float)
#         print("train", x)
        y= y.to(device, dtype=torch.long)

        y= y.squeeze(0)

        output= model(x)
        loss= loss_func(output, y)

        loss.backward()
        model.store_IE_gradient()
        model.optimiser.step()

        losses += [loss.item()]
        outputs += [output.argmax(dim=1).cpu().detach().numpy()]
        y_real += [y.cpu().detach().numpy()]

    model.update_path()
    pickle.dump( model.return_IE_grad(), open(f"{model.path}.pkl", "wb") )
    model.IE_grad=[]
    if len(losses)==1:
        return losses[0], outputs[0], y_real[0]
    else:
        return np.array(losses).mean(), np.concatenate(outputs), np.concatenate(y_real)

def grad_train_rnn(model, dataloader, loss_func, device):
    model.train()

    losses= []
    outputs= []
    y_real= []

    t=tqdm(dataloader, desc="Batch")
    # t=dataloader

    for x, y in t:
        model.optimiser.zero_grad()

        x= x.to(device, dtype=torch.float)
#         print("train", x)
        y= y.to(device, dtype=torch.long)

        y= y.squeeze(0)

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

def train(model, dataloader, loss_func, device):
    model.train()

    losses= []
    outputs= []
    y_real= []

    # t=tqdm(dataloader, desc="Batch")
    t= dataloader

    for x, y in t:
        model.optimiser.zero_grad()

        x= x.to(device, dtype=torch.float)
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

def train_rnn(model, dataloader, loss_func, device):
    model.train()

    losses= []
    outputs= []
    y_real= []

    t=tqdm(dataloader, desc="Batch")

    for x, y in t:
        model.optimiser.zero_grad()

        x= x.to(device, dtype=torch.float)
#         print("train", x)
        y= y.to(device, dtype=torch.long)

        y= y.squeeze(0)

        output= model(x)
        # print("output", output.shape, ", y", y.shape)
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


def val(model, val_loader, loss_func, device):
    model.eval()
#     loss_avg = 0
#     counter = 0
    y_real= []
    outputs= []
    losses= []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device, dtype=torch.float)
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

def val_rnn(model, val_loader, loss_func, device):
    model.eval()
#     loss_avg = 0
#     counter = 0
    y_real= []
    outputs= []
    losses= []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device, dtype=torch.float)
#             print("val", x)
            y = y.to(device, dtype=torch.long)

            y= y.squeeze(0)

            output = model(x)
            loss= loss_func(output, y)

            losses += [loss.item()]
            outputs += [output.argmax(dim=1).cpu().detach().numpy()]
            y_real += [y.cpu().detach().numpy()]
    if len(losses)==1:
        return losses[0], outputs[0], y_real[0]
    else:
        return np.array(losses).mean(), np.concatenate(outputs), np.concatenate(y_real)

def prediction(model, loader, device, dict_flag=False):
    model.eval()
    predictions= []
    y_real= []

    with torch.no_grad():
        for x, y in loader:
#             print(x)
            x= x.to(device, dtype=torch.float)
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

def prediction_rnn(model, loader, device, dict_flag=False):
    model.eval()
    predictions= []
    y_real= []

    with torch.no_grad():
        for x, y in loader:
#             print(x)
            x= x.to(device, dtype=torch.float)
#             x= x.to(device, dytpe=torch.float, non_blocking=False, copy=False, memory_format=torch.preserve_format)
            y= y.squeeze(0)

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


def training(model, loader, train_func, loss_func, device):
### TRAINING
        loss, o, y_real= train_func(model, loader, loss_func, device)
#         print("o", o)
#         print("y_real", y_real)
        dictionary= classification_report(y_real, o, output_dict=True, zero_division=0)

        return loss, dictionary

def validation_rnn(model, loader, loss_func, device):
    loss, o, y_real= val_rnn(model, loader, loss_func, device)
    dictionary= classification_report(y_real, o, output_dict=True, zero_division=0)

    return loss, dictionary

def validation(model, loader, loss_func, device):
    loss, o, y_real= val(model, loader, loss_func, device)
    dictionary= classification_report(y_real, o, output_dict=True, zero_division=0)

    return loss, dictionary
