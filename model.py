'''
This file contains the definition of the neural network that should synthesize the barrier certificates
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lib.utilities import MSE,real_to_complex,complex_to_real,real_to_complex_vectorized,complex_to_real_vectorized

class FullyConnectedNet(nn.Module):
    def __init__(self, input_size: int, num_layer: int, neuron_per_layer: int):
        super(FullyConnectedNet, self).__init__()

        if num_layer < 1:
            raise ValueError("number of hidden layers must be at least 1.")
        if neuron_per_layer < 1:
            raise ValueError("number of neuron per layers must be at least 1.")
        if input_size < 1:
            raise ValueError("input size must be at least 1.")

        # Layer di input (da input_size a n neuroni)
        self.input_layer = nn.Linear(input_size, neuron_per_layer)

        # Layer nascosti
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layer - 1): # num_layer-1 perché il primo layer nascosto è l'input_layer
            self.hidden_layers.append(nn.Linear(neuron_per_layer, neuron_per_layer))

        # Layer di output (da n a 1 neurone)
        self.output_layer = nn.Linear(neuron_per_layer, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        x = self.output_layer(x)
        return x
    
def custom_loss_vectorized(model : FullyConnectedNet, Z0_samples,ZU_samples,other_samples, gamma ,eta, dynamic):
    #add a loss for each value of the matrices that exceeds 1
    Z0 = torch.as_tensor(Z0_samples, dtype=torch.float32)
    ZU = torch.as_tensor(ZU_samples, dtype=torch.float32)
    Other = torch.as_tensor(other_samples, dtype=torch.float32)
    
    device = next(model.parameters()).device
    Z0 = Z0.to(device)
    ZU = ZU.to(device)
    Other = Other.to(device)

    '''Z0 losses'''
    predictions_Z0 = model(Z0)
    target_Z0 = torch.full_like(predictions_Z0, -eta)
    Z0_losses_individual = MSE(predictions_Z0, target_Z0).squeeze()

    '''ZU losses'''
    predictions_ZU = model(ZU)
    target_ZU = torch.full_like(predictions_ZU, eta)
    ZU_losses_individual = MSE(predictions_ZU, target_ZU).squeeze()

    '''Dynamic losses'''
    All_samples = torch.cat([Z0, Other], dim=0)
    predictions_Other = model(Other) 
    predictions_All = torch.cat([predictions_Z0, predictions_Other], dim=0)
    
    f_All = complex_to_real_vectorized(dynamic(real_to_complex_vectorized(All_samples)))
    
    B_f_All = model(f_All)
    target_Dynamic = torch.full_like(B_f_All, -eta)
    
    All_Dynamic_losses_individual = MSE(B_f_All, target_Dynamic).squeeze()
    
    mask = (predictions_All <= gamma).squeeze()
    Dynamic_losses_masked = All_Dynamic_losses_individual[mask]
    
    sum_Z0_loss = torch.sum(Z0_losses_individual)
    sum_ZU_loss = torch.sum(ZU_losses_individual)
    sum_Dynamic_loss = torch.sum(Dynamic_losses_masked)

    total_loss = sum_Z0_loss + sum_ZU_loss + sum_Dynamic_loss
    
    return (
        total_loss, 
        sum_Z0_loss.item() if sum_Z0_loss.numel() > 0 else 0.0,
        sum_ZU_loss.item() if sum_ZU_loss.numel() > 0 else 0.0,
        sum_Dynamic_loss.item() if sum_Dynamic_loss.numel() > 0 else 0.0
    )

def custom_loss(model : FullyConnectedNet, Z0_samples,ZU_samples,other_samples, gamma ,eta, dynamic):
    '''
    this is the loss function for the training
    '''
    individual_losses = []
    Z0_losses = []
    ZU_losses = []
    Dynamic_losses = []
    for z0 in Z0_samples:
        input_tensor = torch.tensor(z0).float().unsqueeze(0)
        prediction=model(input_tensor)
        Z0_losses.append(MSE(prediction,-eta))

        f_z0= complex_to_real(dynamic(real_to_complex(z0)))
        B_f_z0 = model(torch.tensor(f_z0).float().unsqueeze(0))
        
        if prediction<=gamma:
            Dynamic_losses.append(MSE(B_f_z0,-eta))
        
        
    for zu in ZU_samples:
        input_tensor = torch.tensor(zu).float().unsqueeze(0)
        prediction=model(input_tensor)
        ZU_losses.append(MSE(prediction,eta))

    for z in other_samples:
        f_z= complex_to_real(dynamic(real_to_complex(z)))
        B_f_z = model(torch.tensor(f_z).float().unsqueeze(0))

        input_tensor = torch.tensor(z).float().unsqueeze(0)
        prediction=model(input_tensor)

        if prediction<=gamma:
            Dynamic_losses.append(MSE(B_f_z,-eta))

    individual_losses = Dynamic_losses + Z0_losses + ZU_losses
    if individual_losses: 
        stacked_losses = torch.stack(individual_losses)
        total_loss = torch.sum(stacked_losses)
    else:
        total_loss = torch.tensor(0.0, dtype=torch.float32) 
    return total_loss,np.sum([i.item() for i in Z0_losses]),np.sum([i.item() for i in ZU_losses]),np.sum([i.item() for i in Dynamic_losses])

def normalize_sample_set(samples):
    n = len(samples[0])
    avg_features = [0 for _ in range(n)]
    stdev_features = [0 for _ in range(n)]

    for point in samples:
        for i in range(n):
            avg_features[i]+= point[i]
    for point in samples:
        for i in range(n):
            stdev_features[i]+= ((point[i]-avg_features[i])**2)/(n-1)
    
    stdev_features=np.sqrt(np.array(stdev_features))
    avg_features=np.array(avg_features)/n

    normalized_samples=[]
    for point in samples:
        normalized_samples.append(np.array(point-avg_features)/stdev_features)
    return normalized_samples,avg_features,stdev_features


    