import cirq
import openfermion as of
import numpy as np
import datetime
from anzats import Anzats, AnzatsAFMHeisenberg, AnzatsToricCode, AnzatsBCSHubbard
from expectation import AFMHeisenbergArgs
import qsimcirq
# import cupy as cp
# from expectation import get_expectation_ZiZj, get_expectation_ghz_l4, get_expectation_ghz_l8
# from optimization import get_gradient, optimize_by_gradient_descent
from inner_functions import _get_one_body_term_on_hubbard, _get_two_body_term_on_hubbard, _exponentiate_quad_ham


def get_exact_expectation_afm_heisenberg(length):
    # open boundary

    ham = of.ops.QubitOperator()
    for i in range(length-1):
        ham += of.ops.QubitOperator(((i, "X"), (i+1, "X")))
        ham += of.ops.QubitOperator(((i, "Y"), (i+1, "Y")))
        ham += of.ops.QubitOperator(((i, "Z"), (i+1, "Z")))

    sparse_ham = of.linalg.get_sparse_operator(ham)

    energy, state = of.linalg.get_ground_state(
        sparse_ham, initial_guess=None
    )

    return energy, state 


def main():
    print('|length|energy|energy/L|')
    print('|------|------|--------|')
    for length in range(2,20,2):
        # length = 16
        energy, state = get_exact_expectation_afm_heisenberg(length)

        # print(f"state=\n{state}\n")
        print(f'|{length}|{energy}|{energy/length}|')
        # print(f"energy={energy}\n")
        # print(f"energy/L={energy/length}")

if __name__=='__main__':
    main()

