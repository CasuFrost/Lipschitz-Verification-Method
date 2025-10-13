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
from lipschitz_costant import calculate_lipschitz_constant_L2,calculate_lipschitz_constant_Linf

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

num_layer=1
neuron_per_layer=4

model_name = 'models/'+system_name+'/'+str(num_layer)+'_'+str(neuron_per_layer)+'_BEST_MODEL/n'+str(n)+'_eps'+str(epsilon)+'.pth'
model_name='models/z_gate_iter_52/1_4/n4_eps0.1_MINLOSS.pth'

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
gamma=eta
print('eta:',eta)
print('gamma',gamma)

for z in Z0_samples+other_samples:
    pred=model(torch.tensor(z)).item()
    if pred<=gamma:
        f_z = complex_to_real(system.step(real_to_complex(z)))
        B_f_z = model(torch.tensor(f_z).float().unsqueeze(0))
        if B_f_z>-eta:
            errors+=1
            print('dyn error',pred,B_f_z)

print('error in perc.',errors/len(Z0_samples+Z0_samples+other_samples+ZU_samples)*100,'%')
LB=calculate_lipschitz_constant_L2(model)
print('Lipstchiz upper bound in L2:',LB)
print('L_B*epsilon-eta should be less or equal than 0 but is',LB*epsilon-eta)
print('L_B should be at most',eta/epsilon,'but is equal to',LB)