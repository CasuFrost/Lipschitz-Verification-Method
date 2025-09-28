from grid import get_samples
from model import FullyConnectedNet,custom_loss,normalize_sample_set, custom_loss_optimized
from labeling import label_sample_type
from lib.gate import *
from others.semi_algebraic_set import SemiAlgebraicSet
from others.system import System
from lib.plot import plot_scatter_points,plot_losses
import numpy as np
from lib.utilities import generate_var_list
import os
from training import train

import torch
import torch.optim as optim

if __name__ == "__main__":
    '''Optimization and system attributes'''
    eta = 0.1
    gamma = 0.1
    n = 4
    verbose=1
    #num_layer = 1
    #neuron_per_layer = 1
    epsilon = 0.1   
    variables=generate_var_list(n)
    num_epochs = 400
    samples = get_samples(epsilon=epsilon,n=n)

    #normalized_samples,avg_features,stdev_features = normalize_sample_set(samples)
    
    
    
    Z0 = SemiAlgebraicSet(variables,[
        'ReZ1**2 >= 0.9'
    ])

    ZU = SemiAlgebraicSet(variables,[
        'ReZ1**2 + ImZ2**2 <= 0.5'
    ])

    system = System(1,Z_gate(),Z0,ZU,'z_gate_newLoss')


    for num_layer in [1,2,3,4]:
        for neuron_per_layer in [2,3,5,10]:
            model = FullyConnectedNet(n,num_layer,neuron_per_layer)
            data_file,packed_losses=train(system,num_layer,neuron_per_layer,n,epsilon, samples, verbose, optim, model,num_epochs,gamma,eta , loss_function=custom_loss)
            plot_losses(data_file,neuron_per_layer,num_layer,packed_losses)



    
