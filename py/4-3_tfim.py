import os
import datetime
from functools import partial
import tomllib
import multiprocessing as mp

import cirq
import openfermion
import numpy as np
from anzats import Anzats
from expectation import get_expectation_critical_state, get_expectation_critical_state_gpu, TFIMStateArgs
from optimization import optimize_by_gradient_descent, optimize_by_gradient_descent_gpu, optimize_by_gradient_descent_multiprocess

Pi=3.1415


def main():
    with open(".toml", mode="rb") as f:
        config = tomllib.load(f)
        print(config)
    length_list = config["critical_state"]["length_list"]
    p_list = config["critical_state"]["p_list"]
    g = config["critical_state"]["g"]
    alpha = config["critical_state"]["alpha"]
    delta_gamma = config["critical_state"]["delta_gamma"]
    delta_beta = config["critical_state"]["delta_beta"]
    iteration = config["critical_state"]["iteration"]
    gpu = config["critical_state"]["gpu"]
    output_file_prefix = "4-3_tfim"

        
    results_dir_path = config["critical_state"]["results_dir_path"]
    # results_dir_path = os.path.join('.results')
    if not os.path.exists(results_dir_path):
        os.mkdir(results_dir_path)
    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, 'JST')
    now = datetime.datetime.now(JST)
    ymdhms = now.strftime('%Y-%m-%d_%H-%M-%S')



    # too much processes may cause delay because of communication between them
    pool = mp.Pool(2)

    for p in p_list:
        initial_gamma = np.array([0.5 for i in range(p)])
        initial_beta  = np.array([0.5 for i in range(p)])
        for length in length_list:
            qsim_option = {'t': int(length/2), 'f':1}
            csvpath = os.path.join(results_dir_path, '{}_l{:02}_p{}_{}.csv'.format(output_file_prefix, length, p, ymdhms))
            tomlpath = os.path.join(results_dir_path, '{}_l{:02}_p{}_{}.toml'.format(output_file_prefix, length, p, ymdhms))

            
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

            function_args = TFIMStateArgs(length, g, qsim_option)

            print("exact energy: ")
            print(-1.28*length)

            gamma, beta = optimize_by_gradient_descent_multiprocess(
                function=partial(get_expectation_critical_state_gpu, function_args=function_args), 
                initial_gamma=initial_gamma, 
                initial_beta=initial_beta, 
                alpha=alpha, 
                delta_gamma=delta_gamma, 
                delta_beta=delta_beta, 
                iteration=iteration, 
                figure=True,
                filepath=csvpath, 
                pool=pool)
            print(gamma, beta)

    pool.close()
    pool.join()

if __name__=='__main__':
    main()