import os
import shutil
import datetime
import time
from functools import partial
import tomllib
import multiprocessing as mp
import cirq #state vector
import openfermion
import numpy as np
from anzats import Anzats
from expectation import get_expectation_afm_heisenberg_lattice, AFMHeisenbergLatticeArgs
from optimization import optimize_by_gradient_descent_multiprocess, optimize_by_gradient_descent, optimize_by_lbfgsb
    
def main():    
    output_file_prefix = "afm-heisenberg-lattice"
    
    with open(".toml", mode="rb") as f:
        config = tomllib.load(f)

    length_list = config[output_file_prefix]["length_list"]
    p_list = config[output_file_prefix]["p_list"]
    rows_list = config[output_file_prefix]["rows_list"]
    boundary_condition = config[output_file_prefix]["boundary_condition"]
    results_dir_path = config[output_file_prefix]["results_dir_path"]    
    
    if boundary_condition == "PBC":
        periodic = True
    elif boundary_condition == "OBC":
        periodic = False
    else:
        periodic = False
        print(f'{boundary_condition} not valid boundary condition, using OBC.')
    
    if not os.path.exists(results_dir_path):
        os.mkdir(results_dir_path)
        print(f"Directory {results_dir_path} created.")
    else:
        shutil.rmtree(results_dir_path)
        os.mkdir(results_dir_path)
        print(f"Directory {results_dir_path} cleared.")

    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, 'JST')
    now = datetime.datetime.now(JST)
    ymdhms = now.strftime('%Y-%m-%d_%H-%M-%S')
    
    # Start time
    start_time = time.time()
    
    print('Running Scipy optimizer')
    # Run for p and l
    for p in p_list:
        initial_gamma = np.array([0.6 for _ in range(p)])
        initial_beta = np.array([0.6 for _ in range(p)])
        
        for length in length_list:
            qsim_option = {'t': int(length / 2), 'f': 1}
            csvpath = os.path.join(results_dir_path, '{}_l{:02}_p{}_{}.csv'.format(output_file_prefix, length, p, ymdhms))
            tomlpath = os.path.join(results_dir_path, '{}_l{:02}_p{}_{}.toml'.format(output_file_prefix, length, p, ymdhms))

            with open(tomlpath, mode='a') as f:
                f.write("length       ={}\n".format(length))
                f.write("p            ={}\n".format(p))
                f.write("initial_gamma={}\n".format("[" + ", ".join(str(value) for value in initial_gamma.tolist()) + "]"))
                f.write("initial_beta ={}\n".format("[" + ", ".join(str(value) for value in initial_beta.tolist()) + "]"))

            function_args = AFMHeisenbergLatticeArgs(int(length/rows_list[0]), rows_list[0], periodic, qsim_option)

            gamma, beta = optimize_by_lbfgsb(
                    function=partial(get_expectation_afm_heisenberg_lattice, function_args=function_args),
                    initial_gamma=initial_gamma,
                    initial_beta=initial_beta,
                    bounds=None,#[(0, 1)] * (2 * p),
                    figure=True,
                    filepath=csvpath)
                
    # End time
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Write elapsed time to a file
    elapsed_time_file = os.path.join(results_dir_path, '{}_elapsed_time_{}.txt'.format(output_file_prefix, ymdhms))
    with open(elapsed_time_file, mode='w') as f:
        f.write("Elapsed time: {:.2f} seconds".format(elapsed_time))

if __name__ == '__main__':
    main()