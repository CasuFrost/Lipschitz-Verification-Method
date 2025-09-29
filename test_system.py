from grid import get_samples
from model import FullyConnectedNet,custom_loss,normalize_sample_set
from labeling import label_sample_type
from lib.gate import *
from others.semi_algebraic_set import SemiAlgebraicSet
from others.system import System
from lib.plot import plot_scatter_points,plot_losses
import numpy as np
from lib.utilities import generate_var_list
import os
from training import train
from lib.utilities import MSE,real_to_complex,complex_to_real
import torch
import torch.optim as optim

eta = 0.026112914085388184
gamma = 0.026112914085388184
n = 4
verbose=1

epsilon = 0.1   
variables=generate_var_list(n)
num_epochs = 400
samples = get_samples(epsilon=epsilon,n=n)
Z0 = SemiAlgebraicSet(variables,[
    'ReZ1**2 >= 0.9'
])

ZU = SemiAlgebraicSet(variables,[
    'ReZ1**2 + ImZ2**2 <= 0.5'
])

system = System(1,Z_gate(),Z0,ZU,'z_gate_newLoss')
Z0_samples,ZU_samples,other_samples=label_sample_type(samples,system,epsilon, verbose=verbose)
system_name='z_gate_newLoss'

num_layer=2
neuron_per_layer=5

model_name = 'models/'+system_name+'/'+str(num_layer)+'_'+str(neuron_per_layer)+'_BEST_MODEL/n'+str(n)+'_eps'+str(epsilon)+'.pth'

model = FullyConnectedNet(n,num_layer,neuron_per_layer)
checkpoint = torch.load(model_name, weights_only=True) 
model.load_state_dict(checkpoint['model_state_dict']) 

errors=0

max_b = -float('inf')
for p in Z0_samples:
    pred=model(torch.tensor(p)).item()
    max_b = max(max_b,pred)
    if pred>=0:
        errors+=1
        print(pred)

min_b=float('inf')
for p in ZU_samples:
    pred=model(torch.tensor(p)).item()
    min_b=min(pred,min_b)
    if pred<0:
        errors+=1
        print(pred)

eta=min(-max_b,min_b)
print(eta)
gamma=0.02443191409111022

for z in Z0_samples+other_samples:
    pred=model(torch.tensor(z)).item()
    if pred<=gamma:
        f_z = complex_to_real(system.step(real_to_complex(z)))
        B_f_z = model(torch.tensor(f_z).float().unsqueeze(0))
        if B_f_z>-eta:
            errors+=1
            print('dyn error',pred,B_f_z)

print('error in perc.',errors/len(Z0_samples+Z0_samples+other_samples+ZU_samples)*100,'%')