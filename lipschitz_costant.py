import torch
import model
import numpy as np  

def calculate_lipschitz_constant_L2(model):
    L=1
    for name, param in model.named_parameters():
        if 'weight' in name:
            weight_matrix = param.data
            weight_matrix_np = weight_matrix.numpy()
            weight_matrix_np_T = weight_matrix_np.T
            M = weight_matrix_np_T @ weight_matrix_np
            eigenvalues,_ = np.linalg.eig(M)
            max_eg = max(eigenvalues)
            spectral_norm = np.sqrt(max_eg)
            L=L*spectral_norm
    return L

def calculate_lipschitz_constant_Linf(model):
    L=1
    for name, param in model.named_parameters():
        if 'weight' in name:
            weight_matrix = param.data
            weight_matrix_np = weight_matrix.numpy()
            max_row_sum = 0
            for row in weight_matrix_np:
                tmp=0
                for v in row:
                    tmp+=np.abs(v)
                max_row_sum=max(max_row_sum,tmp)
            L=L*max_row_sum
    return L