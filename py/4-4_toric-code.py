import os
import datetime
from functools import partial
import tomllib
import multiprocessing as mp

import cirq
import openfermion
import numpy as np
from anzats import Anzats
from expectation import ToricCodeArgs, get_expectation_toric_code
from optimization import optimize_by_gradient_descent_multiprocess, optimize_by_gradient_descent


def main():
    with open(".toml", mode="rb") as f:
        config = tomllib.load(f)
        print(config)
    length_list = config["toric_code"]["length_list"]
    p_list = config["toric_code"]["p_list"]
    alpha = config["toric_code"]["alpha"]
    delta_gamma = config["toric_code"]["delta_gamma"]
    delta_beta  =config["toric_code"]["delta_beta"]
    iteration = config["toric_code"]["iteration"]

    results_dir_path = config["toric_code"]["results_dir_path"]

    if not os.path.exists(results_dir_path):
        os.mkdir(results_dir_path)

    output_file_prefix = "4-5_afm-heisenberg"
    
    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, 'JST')
    now = datetime.datetime.now(JST)
    ymdhms = now.strftime('%Y-%m-%d_%H-%M-%S')

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
                f.write("alpha        ={}\n".format(alpha))
                f.write("initial_gamma={}\n".format("["+", ".join(str(value) for i, value in enumerate(initial_gamma.tolist()))+"]"))
                f.write("initial_beta ={}\n".format("["+", ".join(str(value) for i, value in enumerate(initial_beta.tolist())) +"]"))
                f.write("delta_gamma  ={}\n".format(delta_gamma))
                f.write("delta_beta   ={}\n".format(delta_beta))
                f.write("iteration    ={}\n".format(iteration))

            function_args = ToricCodeArgs(length, qsim_option)

            gamma, beta = optimize_by_gradient_descent_multiprocess(
                function=partial(get_expectation_toric_code, function_args=function_args), 
                initial_gamma=initial_gamma, 
                initial_beta=initial_beta, 
                alpha=alpha, 
                delta_gamma=delta_gamma, 
                delta_beta=delta_beta, 
                iteration=iteration, 
                figure=True,
                filepath=csvpath)
            print(gamma, beta)

    pool.close()
    pool.join()
if __name__=='__main__':
    main()