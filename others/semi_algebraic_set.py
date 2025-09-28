'''
This file contains a class that models a semi-algebraic real set with a list 
of polynomial constraint
'''

class SemiAlgebraicSet:
    variables = []
    constraints = []
    def __init__(self,var,cons):
        self.variables=var
        self.constraints=cons
    def print(self):
        print("-"*10)
        print("|variables:")
        for v in self.variables:
            print("| ",v)
        print("|constraints:")
        for c in self.constraints:
            print("| ",c)
        print("-"*10,"\n")