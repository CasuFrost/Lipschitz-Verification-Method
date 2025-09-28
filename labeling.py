HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKCYAN = '\033[96m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

import threading
import numpy as np
from others.system import System
from lib.utilities import file_to_array,array_to_file
from others.check_set_emptyness import check_set
from others.semi_algebraic_set import SemiAlgebraicSet
import os
error : float = 1e-6

def label_sample_type(samples, system : System, epsilon,verbose=0):

    '''
    This function, given a set of samples, and a quantum system, label the type of the samples, 
    checking if a point is a Zu-type or Z0-type
    '''
    n = 2**(system.qubit_num+1)
    path = 'samples/n_'+str(n)+'epsilon_'+str(epsilon)

    if os.path.exists(path+'/Z0.txt') and os.path.exists(path+'/ZU.txt') and os.path.exists(path+'/others.txt'):
        return file_to_array(path+'/Z0.txt'),file_to_array(path+'/ZU.txt'),file_to_array(path+'/others.txt')

    
    len_samples = len(samples)
    Z0_type_samples = []
    ZU_type_samples = []
    other = []
    cnt = int(len_samples*0.01)

    for index,p in enumerate(samples):
        if check_point_type(p,system.Z0,epsilon):
            Z0_type_samples.append(p)
        elif check_point_type(p,system.ZU,epsilon):
            ZU_type_samples.append(p)
        else :
            other.append(p)
        if verbose and cnt >= int(len_samples*0.01):
            cnt=0
            os.system("clear")
            print("Labeling "+str(len_samples)+" points..."+ BOLD,str(int(index/len_samples*100))+"%"+ENDC)
        cnt+=1
    

    array_to_file(Z0_type_samples,path,"Z0.txt")
    array_to_file(ZU_type_samples,path,"ZU.txt")
    array_to_file(other,path,"others.txt")
    return Z0_type_samples,ZU_type_samples,other

def label_sample_type_parallel(samples, system : System, epsilon,verbose=0):
    pass


def check_point_type(point, set : SemiAlgebraicSet, epsilon):
    set_to_check = SemiAlgebraicSet(set.variables.copy(),set.constraints.copy())
    sphere_constr = ''
    for i in range(len(set.variables)):
        new_cons_1 = set.variables[i]+' >= '+str(point[i]-epsilon)
        new_cons_2 = set.variables[i]+' <= '+str(point[i]+epsilon)
        set_to_check.constraints.append(new_cons_1)
        set_to_check.constraints.append(new_cons_2)
        sphere_constr = sphere_constr + set.variables[i]+'**2 + '
    set_to_check.constraints.append(sphere_constr[:-2]+'>= '+str(1-error))
    set_to_check.constraints.append(sphere_constr[:-2]+'<= '+str(1+error))
    return check_set(set_to_check)
 



