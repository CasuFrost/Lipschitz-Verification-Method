'''
This file contains a class that models a quantum circuit
A system is defined by:
    a quantum gate (the dynamic of the system)
    the number of qubit involved
    a semi algebraic set defining the initial region
    a semi algebraic set defining the unsafe region
'''
import torch
from lib.gate import *
from others.semi_algebraic_set import SemiAlgebraicSet
class System:
    name = ""
    qubit_number : int = 0
    Z0 : SemiAlgebraicSet 
    ZU : SemiAlgebraicSet
    gate = None

    def __init__(self,qubit_num : int, gate ,Z0 : SemiAlgebraicSet, ZU : SemiAlgebraicSet ,name):
        self.qubit_num=qubit_num
        self.gate=gate 
        self.Z0 = Z0
        self.ZU = ZU
        self.name=name

    def step(self,z):
        if type(z)!=torch.Tensor:
            if self.qubit_num>1:
                dynamic = tensor_gate(self.gate,self.qubit_num)
                return list(np.dot(dynamic,z))
            return list(np.dot(self.gate,z))
        if self.qubit_num > 1:
            evolution_matrix = tensor_gate(self.gate, self.qubit_num) 
        else:
            evolution_matrix = self.gate
        evolution_matrix = torch.as_tensor(self.gate, dtype=torch.complex64)
        evolution_matrix = evolution_matrix.to(z.device)
        return z @ evolution_matrix.transpose(-2, -1) 