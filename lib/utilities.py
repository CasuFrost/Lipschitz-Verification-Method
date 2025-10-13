import numpy as np
import os

'''Macro per i colori in console'''
HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKCYAN = '\033[96m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

def generate_var_list(n):
    res=[]
    n=n//2
    for i in range(n):
        res.append('ReZ'+str(i+1))
        res.append('ImZ'+str(i+1))
    return res

def MSE(x,y):
    return 0.5*((x-y)**2)

def real_to_complex(x):
    z = []
    for i in range(0,len(x),2):
        z.append(complex(x[i],x[i+1]))
    return z
def complex_to_real(z : complex):
    x = []
    for zi in z:
        x.append(zi.real)
        x.append(zi.imag)
    return x

def file_to_array(path):
    array = []
    with open(path, 'r') as f:
        for line in f:
            line = line[:-1].strip("[]")
            line_components = [elem.strip() for elem in line.split(',')]
            array.append([float(elem) for elem in line_components])
    return array

def array_to_file(array,path,filename):
    os.makedirs(path, exist_ok=True)
    with open(path+'/'+filename, 'w') as f:
        for index,elem in enumerate(array):
            f.write(str(elem))
            if index != len(array)-1:
                f.write("\n")


import torch

def real_to_complex_vectorized(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] % 2 != 0:
        raise ValueError("Il numero di feature deve essere pari per la conversione complex_to_real.")
    reshaped_x = x.view(x.shape[0], -1, 2)
    real_part = reshaped_x[..., 0] # (B, N)
    imag_part = reshaped_x[..., 1] # (B, N)
    z = torch.complex(real_part, imag_part)
    return z
import torch

def complex_to_real_vectorized(z: torch.Tensor) -> torch.Tensor:
    real_part = torch.real(z)
    imag_part = torch.imag(z)
    stacked = torch.stack((real_part, imag_part), dim=-1)

    x = stacked.reshape(z.shape[0], -1)
    
    return x


def pretty_print_training_status(iter,num_layer,neuron_per_layer,current_training_session,total_training_session):
    os.system('clear')
    print(BOLD+'Train n. '+str(current_training_session)+' out of '+str(total_training_session)+' --> '
          +str(round(current_training_session/total_training_session*100,2))+'% done.'+ENDC)
    print('     Current iteration:             ',HEADER,iter,ENDC)
    print('     num layer of curr model:       ',HEADER,num_layer,ENDC)
    print('     neuron per layer of curr model:',HEADER,neuron_per_layer,ENDC)
