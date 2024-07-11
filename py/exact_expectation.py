import cirq
import openfermion as of
import numpy as np
import datetime
from anzats import Anzats, AnzatsAFMHeisenberg
from expectation import AFMHeisenbergArgs
import qsimcirq

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

