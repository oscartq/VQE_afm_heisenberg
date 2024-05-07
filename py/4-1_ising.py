import os 
import sys
import datetime
from functools import partial

import cirq
import openfermion
import numpy as np
from anzats import Anzats
from expectation import *
from optimization import get_gradient, optimize_by_gradient_descent, optimize_by_gradient_descent_multiprocess

def main():
    length_list = [8]
    p_list = [4]
    alpha = 0.001
    delta_gamma = 0.001
    delta_beta  = 0.001
    iteration = 100
    results_dir_path = os.path.join('.results')
    if not os.path.exists(results_dir_path):
        os.mkdir(results_dir_path)

    for p in p_list:
        initial_gamma = np.array([0.6 for i in range(p)])
        initial_beta  = np.array([0.6 for i in range(p)])
        # initial_gamma = np.array([0.3338298499584198,0.5585269927978516,0.5999090075492859,0.5951893627643585,0.6022665202617645,0.6437351107597351,0.69035205245018,0.7194274961948395])
        # initial_beta  = np.array([0.7192483842372894,0.6903170347213745,0.6435436308383942,0.602260410785675,0.5952440500259399,0.6001113653182983,0.5583064556121826,0.33388543128967285])
        for length in length_list:
            # length = 10
            # p = 8
            
            t_delta = datetime.timedelta(hours=9)
            JST = datetime.timezone(t_delta, 'JST')
            now = datetime.datetime.now(JST)
            ymdhms = now.strftime('%Y-%m-%d_%H-%M-%S')
            csvpath = os.path.join(results_dir_path, '4-1_ising_l{:02}_p{}_{}.csv'.format(length, p, ymdhms))
            tomlpath = os.path.join(results_dir_path, '4-1_ising_l{:02}_p{}_{}.toml'.format(length, p, ymdhms))

            with open(tomlpath, mode='a') as f:
                f.write("length       ={}\n".format(length))
                f.write("p            ={}\n".format(p))
                f.write("alpha        ={}\n".format(alpha))
                f.write("initial_gamma={}\n".format("["+", ".join(str(value) for i, value in enumerate(initial_gamma.tolist()))+"]"))
                f.write("initial_beta ={}\n".format("["+", ".join(str(value) for i, value in enumerate(initial_beta.tolist())) +"]"))
                f.write("delta_gamma  ={}\n".format(delta_gamma))
                f.write("delta_beta   ={}\n".format(delta_beta))
                f.write("iteration    ={}\n".format(iteration))

            qsim_option = {'t': 4, 'f':1}
            function_args = GHZStateArgs(length, qsim_option)
            gamma, beta = optimize_by_gradient_descent_multiprocess(partial(get_expectation_ghz, function_args=function_args),initial_gamma,initial_beta,alpha,delta_gamma,delta_beta ,iteration,True,csvpath)
            print(gamma, beta)


if __name__=='__main__':
    main()