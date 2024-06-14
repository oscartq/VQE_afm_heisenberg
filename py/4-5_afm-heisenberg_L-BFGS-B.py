import os
import shutil
import datetime
from functools import partial
import tomllib
import multiprocessing as mp
import cirq
import openfermion
import numpy as np
from anzats import Anzats
from expectation import get_expectation_afm_heisenberg, AFMHeisenbergArgs
from optimization import optimize_by_lbfgsb

def main(): # Main function
    output_file_prefix = "4-5_afm-heisenberg"

    # input a config file
    with open(".toml", mode="rb") as f:
        config = tomllib.load(f)

    length_list = config[output_file_prefix]["length_list"]
    p_list = config[output_file_prefix]["p_list"]
    #iteration = config[output_file_prefix]["iteration"]
    results_dir_path = config[output_file_prefix]["results_dir_path"]
    
    if not os.path.exists(results_dir_path):
        os.mkdir(results_dir_path)
        print(f"Directory {results_dir_path} created.")
    if os.path.exists(results_dir_path):
        shutil.rmtree(results_dir_path)
        os.mkdir(results_dir_path)
        print(f"Directory {results_dir_path} cleared.")

    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, 'JST')
    now = datetime.datetime.now(JST)
    ymdhms = now.strftime('%Y-%m-%d_%H-%M-%S')

    pool = mp.Pool(4)

    # Run for p and l
    for p in p_list:
        initial_gamma = np.array([0.6 for i in range(p)])
        initial_beta = np.array([0.6 for i in range(p)])

        for length in length_list:
            qsim_option = {'t': int(length / 2), 'f': 1}
            csvpath = os.path.join(results_dir_path, '{}_l{:02}_p{}_{}.csv'.format(output_file_prefix, length, p, ymdhms))
            tomlpath = os.path.join(results_dir_path, '{}_l{:02}_p{}_{}.toml'.format(output_file_prefix, length, p, ymdhms))

            with open(tomlpath, mode='a') as f:
                f.write("length       ={}\n".format(length))
                f.write("p            ={}\n".format(p))
                f.write("initial_gamma={}\n".format("[" + ", ".join(str(value) for value in initial_gamma.tolist()) + "]"))
                f.write("initial_beta ={}\n".format("[" + ", ".join(str(value) for value in initial_beta.tolist()) + "]"))
                #f.write("iteration    ={}\n".format(iteration))

            function_args = AFMHeisenbergArgs(length, qsim_option)

            gamma, beta = optimize_by_lbfgsb(
                function=partial(get_expectation_afm_heisenberg, function_args=function_args),
                initial_gamma=initial_gamma,
                initial_beta=initial_beta,
                figure=True,
                filepath=csvpath)

if __name__ == '__main__':
    main()
