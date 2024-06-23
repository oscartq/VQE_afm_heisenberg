import os
import datetime
from functools import partial
import tomllib
import multiprocessing as mp

import cirq
import openfermion
import numpy as np
import scipy
import scipy.optimize 
from autograd import grad, jacobian

def get_norm(param, function_args):
    return np.linalg.norm(param)

def get_derivative(param, functional_args):
    return np.sum(param)

def main():
    # define param
    p=2
    initial_gamma = np.array([0.6 for i in range(p)])
    initial_beta  = np.array([0.6 for i in range(p)])
    param = np.concatenate([initial_gamma, initial_beta])
    function_args = ''
    iteration = 1000
    
    # minimized = scipy.optimize.minimize(
    #     fun=get_norm,
    #     x0=param
    # )

    minimized = scipy.optimize.minimize(
        fun=get_norm,
        x0=param,
        args=(function_args, ),
        # jac=partial(get_gradient_for_scipy, function=get_expectation_afm_heisenberg_with_concated_parameters),
        method='L-BFGS-B',
        options={'maxiter': iteration},
        tol=1e-6,
        )

    print(minimized)

if __name__=='__main__':
    main()