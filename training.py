'''
this file contains the function that starts the training of the model
'''
from grid import get_samples
from model import FullyConnectedNet,custom_loss
from labeling import label_sample_type
from lib.gate import *
from others.semi_algebraic_set import SemiAlgebraicSet
from others.system import System
from lib.plot import plot_scatter_points
import numpy as np
from lib.utilities import generate_var_list
import os
import torch
import torch.optim as optim

def save_model(epoch,model,optimizer,system,num_layer,neuron_per_layer,model_name,min_loss=-1):
    checkpoint = {
                    'min_loss' : min_loss,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
    }
    os.makedirs('models/'+system.name+'/'+str(num_layer)+'_'+str(neuron_per_layer),exist_ok=True)
    torch.save(checkpoint, model_name)

def train(system : System,num_layer,neuron_per_layer,n,epsilon, samples, verbose, optim, model,num_epochs,gamma,eta,loss_function):

    losses = []

    model_name = 'models/'+system.name+'/'+str(num_layer)+'_'+str(neuron_per_layer)+'/n'+str(n)+'_eps'+str(epsilon)+'.pth'

     #return model_name[:-3]+'txt'

    Z0_samples,ZU_samples,other_samples=label_sample_type(samples,system,epsilon, verbose=verbose)
    if verbose:
        os.system('clear')

    epoch_counter=0
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    min_loss=min_loss=float('inf')

    if os.path.exists(model_name):
        checkpoint = torch.load(model_name)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_counter = checkpoint['epoch']
        min_loss=checkpoint['min_loss']
    if min_loss==-1:
        min_loss=float('inf')
    try:
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            packed_losses = loss_function(model,Z0_samples,ZU_samples,other_samples,gamma,eta, system.step)
            tensor_loss = packed_losses[0]
            tensor_loss.backward()
            optimizer.step()
            loss = tensor_loss.item()
            if loss<min_loss:
                save_model(epoch+epoch_counter,model,optimizer,system,num_layer,neuron_per_layer,model_name[:-4]+'_MINLOSS.pth',min_loss)
                min_loss=loss
            losses.append((epoch+epoch_counter,loss,packed_losses[1],packed_losses[2],packed_losses[3]))
            if epoch+epoch_counter>=num_epochs:
                save_model(epoch+epoch_counter,model,optimizer,system,num_layer,neuron_per_layer,model_name,min_loss)
                break
        save_model(epoch+epoch_counter,model,optimizer,system,num_layer,neuron_per_layer,model_name)
    except KeyboardInterrupt:
        save_model(epoch+epoch_counter,model,optimizer,system,num_layer,neuron_per_layer,model_name,min_loss)
        print('\033[91m\nSESSION INTERRUPTED BY THE USER.\033[0m',)
        exit()
    with open(model_name[:-3]+'txt', 'a') as f:
        for epoch,loss,Z0_loss,ZU_loss,dynamic_loss in losses:
            f.write(str(epoch)+' '+str(loss)+' '+str(Z0_loss)+' '+str(ZU_loss)+' '+str(dynamic_loss)+'\n')

    return model_name[:-3]+'txt',[]

