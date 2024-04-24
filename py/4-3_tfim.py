import cirq
import openfermion
import numpy as np
import datetime
from anzats import Anzats
from expectation import get_expectation_critical_state, TFIMStateArgs
from optimization import optimize_by_gradient_descent
from functools import partial
Pi=3.1415


def optimize_tfim(length, p, g):
    initial_gamma = np.array([0.5 for i in range(p)])
    initial_beta  = np.array([0.5 for i in range(p)])
    function_args = TFIMStateArgs(length, g)

    iteration = 10
    alpha = 0.01
    delta_gamma = 0.001
    delta_beta  = 0.001

    # print("exact energy: ")
    # print(-1.28*length)

    gamma, beta = optimize_by_gradient_descent(partial(get_expectation_critical_state, function_args=function_args), initial_gamma, initial_beta, alpha, delta_gamma, delta_beta, iteration)

    print(gamma, beta)

def main():
    length_list = [4,8,10,12,14,16]
    p_list = [1,2,3,4,5,6,7,8,9,10]
    g = 0.6

    for length in length_list:
        for p in p_list:
            optimize_tfim(length, p, g)

if __name__=='__main__':
    main()