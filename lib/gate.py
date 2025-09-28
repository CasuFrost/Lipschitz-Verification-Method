'''
Porte logiche quantistiche
'''

import numpy as np 
import cmath
import math

inv_sqrt2  = 1/(np.sqrt(2))

def tensor_gate(gate,n): #esegue il prodotto tensore n-esimo di una matrice per se stessa
    if n == 1:
        return gate
    elif n == 2:
        return np.kron(gate, gate)
    else:
        risultato = np.kron(gate, gate)
        for _ in range(n - 2):
            risultato = np.kron(risultato, gate)
        return risultato
    
def CZ_gate():
    return [[1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,-1]]

def X_ROT(theta):
    return [[np.cos(theta/2), complex(0,-np.sin(theta/2))],
            [complex(0,-np.sin(theta/2)), np.cos(theta/2)]]

def Z_ROT(theta):
    return [[1,0],
            [0,complex(np.cos(theta),np.sin(theta))]]

def X_gate():
    return [[0,1],[1,0]]

def Z_gate(): 
    return [[1,0],[0,-1]]

def S_gate(): 
    return [[1,0],[0,complex(0,1)]]


def X_gate():
    return [[0,1],[1,0]]

def Hadamard_gate():
    return [[inv_sqrt2 ,inv_sqrt2 ],[inv_sqrt2 ,inv_sqrt2*(-1)]]

def CH_gate(): #per i sistemi con 2 qubit
    return [[1,0,0,0],
            [0,1,0,0],
            [0,0,inv_sqrt2 ,inv_sqrt2 ],
            [0,0,inv_sqrt2 ,inv_sqrt2 *(-1)]]

def T_gate():
    return [[1,0],
            [0,cmath.exp((cmath.pi*complex(0,1))/4)]]

theta = 0
def Y_ROT_gate():
    return [
        [np.cos(theta/2),-np.sin(theta/2)],
        [np.sin(theta/2),np.cos(theta/2)]
    ]


def z_grover(phi):
    return np.array([math.cos(phi),math.sin(phi)])