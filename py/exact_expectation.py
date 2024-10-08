import cirq
import openfermion as of
import numpy as np
from itertools import product

def run_exact_expectation_state(file_prefix, length, width, periodic=True):

    try:
        if file_prefix=='afm-heisenberg':
            return get_exact_expectation_afm_heisenberg(length, width, periodic)
        if file_prefix=='afm-heisenberg-lattice':
            return get_exact_expectation_afm_heisenberg_lattice(length, width, periodic)  
    except ValueError:
        print("input a correct file_prefix: {}".format(file_prefix))

def get_exact_expectation_afm_heisenberg(length, periodic=True):
    # open boundary
    ham = of.ops.QubitOperator()

    edge = 0 if periodic else 1
    for i in range(length-edge):
        ham += of.ops.QubitOperator(((i, "X"), ((i+1)%length, "X")))
        ham += of.ops.QubitOperator(((i, "Y"), ((i+1)%length, "Y")))
        ham += of.ops.QubitOperator(((i, "Z"), ((i+1)%length, "Z")))

    sparse_ham = of.linalg.get_sparse_operator(ham)
    # sparse_ham = of.linalg.generate_linear_qubit_operator(ham)

    energy, state = of.linalg.get_ground_state(
        sparse_ham, initial_guess=None
    )

    return energy, state 

def get_exact_expectation_afm_heisenberg_lattice(rows, cols, periodic=True):
    # open boundary
    ham = of.ops.QubitOperator()

    edge = 1-1 if periodic else 1-0
    # row
    for i in range(rows-edge):
        for j in range(cols):
            current_index = j * rows + i
            right_neighbor = j * rows + (i + 1) % rows
            ham += of.ops.QubitOperator(((current_index, "X"), (right_neighbor, "X")))
            ham += of.ops.QubitOperator(((current_index, "Y"), (right_neighbor, "Y")))
            ham += of.ops.QubitOperator(((current_index, "Z"), (right_neighbor, "Z")))

    # column
    for i in range(rows):
        for j in range(cols-edge):
            current_index = j * rows + i
            down_neighbor = ((j + 1) % cols) * rows + i 
            ham += of.ops.QubitOperator(((current_index, "X"), (down_neighbor, "X")))
            ham += of.ops.QubitOperator(((current_index, "Y"), (down_neighbor, "Y")))
            ham += of.ops.QubitOperator(((current_index, "Z"), (down_neighbor, "Z")))

    sparse_ham = of.linalg.get_sparse_operator(ham)

    energy, state = of.linalg.get_ground_state(
        sparse_ham, initial_guess=None
    )

    return energy, state

def run_expectations_on_heisenberg():
    print('|rows x cols|energy|energy/L|periodic|')
    print('|-----|-----|------|--------|--------|')
    
    rows_list = [2,4]
    
    cols_list  = [2,4]
    particle_n = 8
    periodic = False

    for rows, cols  in product(cols_list, rows_list):
        if rows == 1:
            energy, state = get_exact_expectation_afm_heisenberg(cols, periodic=periodic)
        else:
            energy, state = get_exact_expectation_afm_heisenberg_lattice(rows, cols, periodic=periodic)

        # print(f"state=\n{state}\n")
        print(f'|{rows}x{cols}|{energy}|{energy/rows}|{periodic}|')

def main():
    """outputs exact energies on model parameters
    1. select a function model 
    2. run me like `python exact_expectation.py`
    """
    run_exact_expectation_state()

if __name__=='__main__':
    main()