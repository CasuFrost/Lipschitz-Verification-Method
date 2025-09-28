import os
from z3 import *
import subprocess
from others.semi_algebraic_set import SemiAlgebraicSet
'''
This file contains a function that check if a semi-algebraic sets is empty or not
'''

def generate_SMT_file(variables,constraints):
    file = open("smt_file/tmp.smt2","w+")
    file.write("(set-logic QF_NRA)\n")
    for x in variables:
        file.write('(declare-fun '+x+' () Real)\n')
    for const in constraints:
        ineq_form = const[1]
        polynom = const[0]
        file.write('\n(assert\n    ('+ineq_form)
        file.write("\n      (+\n            ")
        for term in polynom:
            minus=False
            if term[0]=='-':
                minus=True
                term = term[1:]
            if '*' in term:
                tmp = term.split("*")
                term = '( * '+tmp[0]+' '+tmp[1]+')'
            if '^' in term:
                tmp1 = term.split("^")[0]
                tmp2 = term.split("^")[1]
                term = tmp1[:-1]+"( pow "+tmp1[-1]+" "+tmp2[0]+" )"+tmp2[1:]
            if minus:
                tmp = "(* "+term+" -1)"
                term=tmp
            file.write(term+"\n            ")
        file.write("\n      ) 0 \n")
        file.write('    )\n)\n')
    file.write('\n(check-sat)\n(get-model)\n(exit)')
    
def rewrite_pow(s):
    splitted = s.split()
    res = ''
    for sub in splitted:
        if '^' in sub:
            pow = int(sub.split('^')[1])
            radix = sub.split('^')[0]
            new_s = ''
            for _ in range(pow):
                new_s = new_s + radix +'*'
            sub = new_s[:-1]
        res = res+sub+' '
    return res[:-1]

def check_set(set : SemiAlgebraicSet):
    '''
    PARAMETER LIST:
        variables : a list of string, every string is a variable 
        constraint : a list of string defining the constraint of the set
    EXAMPLE: 
        variables = ['x','y','z','w']

        every constraint is a string defining the polynom as follows:
                '2*x + x^2 + z^2 - 5*w - 9 >= 0'
    '''
    s = Solver()
    VAR = {}
    for v in set.variables:
        VAR[v]=Real(v)

    for constr in set.constraints:
        for v in set.variables:
            if v in constr:
                constr = constr.replace(v,'VAR["'+v+'"]')
        s.add(eval(constr))
    if s.check()==sat:
        return True
    return False