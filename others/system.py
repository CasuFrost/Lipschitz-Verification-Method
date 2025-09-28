'''
This file contains a class that models a quantum circuit
A system is defined by:
    a quantum gate (the dynamic of the system)
    the number of qubit involved
    a semi algebraic set defining the initial region
    a semi algebraic set defining the unsafe region
'''
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
        if self.qubit_num>1:
            dynamic = tensor_gate(self.gate,self.qubit_num)
            return list(np.dot(dynamic,z))
        return list(np.dot(self.gate,z))
        