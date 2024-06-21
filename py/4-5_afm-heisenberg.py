import os
import datetime
import time
from functools import partial
import tomllib
import multiprocessing as mp
import cirq
import openfermion
import numpy as np
from anzats import Anzats  # Ensure these imports are correct and available
from expectation import get_expectation_afm_heisenberg, AFMHeisenbergArgs
from optimization import optimize_by_gradient_descent_multiprocess, optimize_by_gradient_descent, optimize_by_lbfgsb

def main(): #Main function
    output_file_prefix = "4-5_afm-heisenberg"

    # input a config file
    with open(".toml", mode="rb") as f:
        config = tomllib.load(f)

    length_list = config[output_file_prefix]["length_list"]
    p_list = config[output_file_prefix]["p_list"]
    alpha = config[output_file_prefix]["alpha"]
    delta_gamma = config[output_file_prefix]["delta_gamma"]
    delta_beta = config[output_file_prefix]["delta_beta"]
    iteration = config[output_file_prefix]["iteration"]
    results_dir_path = config[output_file_prefix]["results_dir_path"]
    if not os.path.exists(results_dir_path):
        os.mkdir(results_dir_path)

    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, 'JST')
    now = datetime.datetime.now(JST)
    ymdhms = now.strftime('%Y-%m-%d_%H-%M-%S')

    pool = mp.Pool(4)

    # Start time
    start_time = time.time()

    # run for p and l
    for p in p_list:
        initial_gamma = np.array([0.6 for _ in range(p)])
        initial_beta = np.array([0.6 for _ in range(p)])
        # initial_gamma = np.array([0.44100484997034073,0.5607438758015633,0.5727711990475655,0.5745645072311163,0.5770069658756256,0.5438836682587862,0.3574042972177267])
        # initial_beta = np.array([0.518215598538518,0.5762445759028196,0.6036094501614571,0.60199218057096,0.5795486941933632,0.5976213440299034,0.5236566364765167])
        for length in length_list:
            qsim_option = {'t': int(length / 2), 'f': 1}
            csvpath = os.path.join(results_dir_path, '{}_l{:02}_p{}_{}.csv'.format(output_file_prefix, length, p, ymdhms))
            tomlpath = os.path.join(results_dir_path, '{}_l{:02}_p{}_{}.toml'.format(output_file_prefix, length, p, ymdhms))

            with open(tomlpath, mode='a') as f:
                f.write("length       ={}\n".format(length))
                f.write("p            ={}\n".format(p))
                f.write("alpha        ={}\n".format(alpha))
                f.write("initial_gamma={}\n".format("[" + ", ".join(str(value) for value in initial_gamma.tolist()) + "]"))
                f.write("initial_beta ={}\n".format("[" + ", ".join(str(value) for value in initial_beta.tolist()) + "]"))
                f.write("delta_gamma  ={}\n".format(delta_gamma))
                f.write("delta_beta   ={}\n".format(delta_beta))
                f.write("iteration    ={}\n".format(iteration))

            function_args = AFMHeisenbergArgs(length, qsim_option)

            gamma, beta = optimize_by_gradient_descent_multiprocess(
                function=partial(get_expectation_afm_heisenberg, function_args=function_args),
                initial_gamma=initial_gamma,
                initial_beta=initial_beta,
                alpha=alpha,
                delta_gamma=delta_gamma,
                delta_beta=delta_beta,
                iteration=iteration,
                figure=True,
                filepath=csvpath,
                pool=pool)

    pool.close()
    pool.join()

    # End time
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Write elapsed time to a file
    elapsed_time_file = os.path.join(results_dir_path, '{}_elapsed_time_{}.txt'.format(output_file_prefix, ymdhms))
    with open(elapsed_time_file, mode='w') as f:
        f.write("Elapsed time: {:.2f} seconds".format(elapsed_time))

if __name__ == '__main__':
    main()