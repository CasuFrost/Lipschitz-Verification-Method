from grid import get_samples
from model import FullyConnectedNet,custom_loss,normalize_sample_set,custom_loss_vectorized
from labeling import label_sample_type
from lib.gate import *
from others.semi_algebraic_set import SemiAlgebraicSet
from others.system import System
from lib.plot import plot_scatter_points,plot_losses
import numpy as np
from lib.utilities import generate_var_list,pretty_print_training_status
import os
from lipschitz_costant import calculate_lipschitz_constant_L2
from training import train

import torch
import torch.optim as optim

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    '''Optimization and system attributes'''
    eta = 0.1
    gamma = 0.1
    n = 4
    verbose=0
    #num_layer = 1
    #neuron_per_layer = 1
    epsilon = 0.1   
    variables=generate_var_list(n)
    num_epochs = 500
    samples = get_samples(epsilon=epsilon,n=n)

    #normalized_samples,avg_features,stdev_features = normalize_sample_set(samples)
    
    
    
    Z0 = SemiAlgebraicSet(variables,[
        'ReZ1**2 >= 0.9'
    ])

    ZU = SemiAlgebraicSet(variables,[
        'ReZ1**2 + ImZ2**2 <= 0.5'
    ])

    iterations=[i+34 for i in range(500)]
    possible_num_layer = [1]
    possible_neuron_per_layer=[4]

    total_training_session = len(iterations)*len(possible_num_layer)*len(possible_neuron_per_layer)
    current_training_session = 1
    #Provare più volte per ogni scelta della struttura della NN
    for iter in iterations:
        system = System(1,Z_gate(),Z0,ZU,'z_gate_iter_'+str(iter))
        for num_layer in possible_num_layer:
            for neuron_per_layer in possible_neuron_per_layer:
                model = FullyConnectedNet(n,num_layer,neuron_per_layer)
                model.to(device)
                pretty_print_training_status(iter,num_layer,neuron_per_layer,current_training_session,total_training_session)
                data_file,packed_losses=train(system,num_layer,neuron_per_layer,n,epsilon, samples, verbose, optim, model,num_epochs,gamma,eta , loss_function=custom_loss_vectorized)
                plot_losses(data_file,neuron_per_layer,num_layer,packed_losses)
                current_training_session+=1

    
