import numpy as np
import os

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
