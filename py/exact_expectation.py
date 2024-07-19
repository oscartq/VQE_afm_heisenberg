import cirq
import openfermion as of
import numpy as np
import datetime
from itertools import product
from anzats import Anzats, AnzatsAFMHeisenberg
from expectation import AFMHeisenbergArgs

def run_exact_expectation_state(file_prefix, length, width, periodic=True):

    try:
        if file_prefix=='4-5_afm-heisenberg':
            return get_exact_expectation_afm_heisenberg(length, periodic)
        if file_prefix=='4-5-2_afm-heisenberg':
            return get_exact_expectation_afm_heisenberg_lattice(length, width, periodic)  
    except ValueError:
        print("input a correct file_prefix: {}".format(file_prefix))
            
def get_exact_expectation_afm_heisenberg(length, periodic=True):
    # open boundary
    ham = of.ops.QubitOperator()

    edge = 1-1 if periodic else 1-0
    for i in range(length-edge):
        ham += of.ops.QubitOperator(((i, "X"), ((i+1)%length, "X")))
        ham += of.ops.QubitOperator(((i, "Y"), ((i+1)%length, "Y")))
        ham += of.ops.QubitOperator(((i, "Z"), ((i+1)%length, "Z")))

    sparse_ham = of.linalg.get_sparse_operator(ham)

    energy, state = of.linalg.get_ground_state(
        sparse_ham, initial_guess=None
    )

    return energy, state 


def get_exact_expectation_afm_heisenberg_lattice(length, width, periodic=True):
    # open boundary
    ham = of.ops.QubitOperator()

    edge = 1-1 if periodic else 1-0
    # row
    for i in range(length-edge):
        for j in range(width):
            current_index = j * length + i
            right_neighbor = j * length + (i + 1) % length

            ham += of.ops.QubitOperator(((current_index, "X"), (right_neighbor, "X")))
            ham += of.ops.QubitOperator(((current_index, "Y"), (right_neighbor, "Y")))
            ham += of.ops.QubitOperator(((current_index, "Z"), (right_neighbor, "Z")))

    # column
    for i in range(length):
        for j in range(width-edge):
            current_index = j * length + i
            down_neighbor = ((j + 1) % width) * length + i 
            ham += of.ops.QubitOperator(((current_index, "X"), (down_neighbor, "X")))
            ham += of.ops.QubitOperator(((current_index, "Y"), (down_neighbor, "Y")))
            ham += of.ops.QubitOperator(((current_index, "Z"), (down_neighbor, "Z")))

    sparse_ham = of.linalg.get_sparse_operator(ham)

    energy, state = of.linalg.get_ground_state(
        sparse_ham, initial_guess=None
    )

    return energy, state 

def main():
    """outputs exact energies on model parameters
    1. select a function model 
    2. run me like `python exact_expectation.py`
    """
    run_expectations_on_heisenberg()
    # run_expectations_on_hubbard()

if __name__=='__main__':
    main()