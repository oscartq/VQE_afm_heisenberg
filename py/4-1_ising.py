import os 
import sys
import datetime

import cirq
import openfermion
import numpy as np
from anzats import Anzats
from expectation import *
from optimization import get_gradient, optimize_by_gradient_descent

def main():
    length_list = [8,10,12,14,16]
    alpha = 0.01
    delta_gamma = 0.001
    delta_beta  = 0.001
    iteration = 30
    results_dir_path = os.path.join('.results')
    if not os.path.exists(results_dir_path):
        os.mkdir(results_dir_path)

    for length in length_list:
        for p in range(1,int(length/2)+1):
            # length = 10
            # p = 5
            initial_gamma = np.array([(0.5) for i in range(p)])
            initial_beta  = np.array([(0.5) for i in range(p)])
            
            t_delta = datetime.timedelta(hours=9)
            JST = datetime.timezone(t_delta, 'JST')
            now = datetime.datetime.now(JST)
            ymdhms = now.strftime('%Y-%m-%d_%H-%M-%S')
            csvpath = os.path.join(results_dir_path, '4-1_ising_l{:02}_p{}_{}.csv'.format(length, p, ymdhms))
            tomlpath = os.path.join(results_dir_path, '4-1_ising_l{:02}_p{}_{}.toml'.format(length, p, ymdhms))

            gamma, beta = optimize_by_gradient_descent(eval("get_expectation_ghz_l{}".format(length)),initial_gamma,initial_beta,alpha,delta_gamma,delta_beta ,iteration,True,csvpath)
            print(gamma, beta)

            with open(tomlpath, mode='a') as f:
                f.write("length       ={}\n".format(length))
                f.write("p            ={}\n".format(p))
                f.write("alpha        ={}\n".format(alpha))
                f.write("initial_gamma={}\n".format(initial_gamma))
                f.write("initial_beta ={}\n".format(initial_beta))
                f.write("delta_gamma  ={}\n".format(delta_gamma))
                f.write("delta_beta   ={}\n".format(delta_beta))
                f.write("iteration    ={}\n".format(iteration))

if __name__=='__main__':
    main()