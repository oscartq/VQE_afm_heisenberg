import os
import datetime
from functools import partial

import cirq
import openfermion
import numpy as np
from anzats import Anzats
from expectation import get_expectation_critical_state, TFIMStateArgs
from optimization import optimize_by_gradient_descent

Pi=3.1415


def optimize_tfim(length, 
                            g,
                            p, 
                            alpha, 
                            delta_gamma, 
                            delta_beta, 
                            iteration, 
                            csvpath, 
                            tomlpath):
    initial_gamma = np.array([0.5 for i in range(p)])
    initial_beta  = np.array([0.5 for i in range(p)])

    function_args = TFIMStateArgs(length, g)

    print("exact energy: ")
    print(-1.28*length)

    gamma, beta = optimize_by_gradient_descent(partial(get_expectation_critical_state, function_args=function_args), initial_gamma, initial_beta, alpha, delta_gamma, delta_beta, iteration, True, csvpath)
    print(gamma, beta)


    with open(tomlpath, mode='a') as f:
        f.write("length       ={}\n".format(length))
        f.write("p            ={}\n".format(p))
        f.write("g            ={}\n".format(g))
        f.write("alpha        ={}\n".format(alpha))
        f.write("initial_gamma={}\n".format("["+", ".join(str(value) for i, value in enumerate(initial_gamma.tolist()))+"]"))
        f.write("initial_beta ={}\n".format("["+", ".join(str(value) for i, value in enumerate(initial_beta.tolist())) +"]"))
        f.write("delta_gamma  ={}\n".format(delta_gamma))
        f.write("delta_beta   ={}\n".format(delta_beta))
        f.write("iteration    ={}\n".format(iteration))

def main():
    # length_list = [4]
    # p_list = [1,2,3]
    length_list = [4,8,10]
    p_list = [1,2,3,4,5,6,7,8,9,10]
    g = 0.8

    alpha = 0.01
    delta_gamma = 0.001
    delta_beta  = 0.001
    iteration = 10

    results_dir_path = os.path.join('.results')
    if not os.path.exists(results_dir_path):
        os.mkdir(results_dir_path)
    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, 'JST')
    now = datetime.datetime.now(JST)
    ymdhms = now.strftime('%Y-%m-%d_%H-%M-%S')


    for length in length_list:
        for p in p_list:
            csvpath = os.path.join(results_dir_path, '4-3_critical_l{:02}_g{:02}_p{}_{}.csv'.format(length, g, p, ymdhms))
            tomlpath = os.path.join(results_dir_path, '4-3_critical_l{:02}_g{:02}_p{}_{}.toml'.format(length, g, p, ymdhms))
            optimize_tfim(length, 
                            g,
                            p, 
                            alpha, 
                            delta_gamma, 
                            delta_beta, 
                            iteration, 
                            csvpath, 
                            tomlpath)

if __name__=='__main__':
    main()