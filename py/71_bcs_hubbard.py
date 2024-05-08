import os
import datetime
from functools import partial
import tomllib
import multiprocessing as mp

import cirq
import openfermion
import numpy as np
from anzats import Anzats
from expectation import get_expectation_bcs_hubbard, HubbardArgs
from optimization import optimize_by_gradient_descent_multiprocess, optimize_by_gradient_descent


def main():
    output_file_prefix = "71_bcs_hubbard"

    # input a config file
    with open(".toml", mode="rb") as f:
        config = tomllib.load(f)

    length_list = config[output_file_prefix]["length_list"]
    width_list = config[output_file_prefix]["width_list"]
    coulomb_list = config[output_file_prefix]["coulomb_list"]
    p_list = config[output_file_prefix]["p_list"]

    alpha = config[output_file_prefix]["alpha"]
    delta_gamma = config[output_file_prefix]["delta_gamma"]
    delta_beta  =config[output_file_prefix]["delta_beta"]
    iteration = config[output_file_prefix]["iteration"]
    results_dir_path = config[output_file_prefix]["results_dir_path"]
    if not os.path.exists(results_dir_path):
        os.mkdir(results_dir_path)


    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, 'JST')
    now = datetime.datetime.now(JST)
    ymdhms = now.strftime('%Y-%m-%d_%H-%M-%S')

    pool = mp.Pool(4)

    # run for p and l
    for p in p_list:
        initial_gamma = np.array([0.2 for i in range(p)])
        initial_beta  = np.array([0.2 for i in range(p)])

        
        for length in length_list:
            for coulomb in coulomb_list:
                qsim_option = {'t': int(length/2), 'f':1}
                csvpath = os.path.join(results_dir_path, '{}_Ut{:02}_l{:02}_p{}_{}.csv'.format(output_file_prefix, coulomb, length, p, ymdhms))
                tomlpath = os.path.join(results_dir_path, '{}_Ut{:02}_l{:02}_p{}_{}.toml'.format(output_file_prefix, coulomb, length, p, ymdhms))

                
                with open(tomlpath, mode='a') as f:
                    f.write("length       ={}\n".format(length))
                    f.write("coulomb      ={}\n".format(coulomb))
                    f.write("p            ={}\n".format(p))
                    f.write("alpha        ={}\n".format(alpha))
                    f.write("initial_gamma={}\n".format("["+", ".join(str(value) for i, value in enumerate(initial_gamma.tolist()))+"]"))
                    f.write("initial_beta ={}\n".format("["+", ".join(str(value) for i, value in enumerate(initial_beta.tolist())) +"]"))
                    f.write("delta_gamma  ={}\n".format(delta_gamma))
                    f.write("delta_beta   ={}\n".format(delta_beta))
                    f.write("iteration    ={}\n".format(iteration))

                function_args = HubbardArgs(
                    x_dimension=width_list[0],
                    y_dimension=length,
                    tunneling=1.,
                    coulomb=coulomb,
                    chemical_potential=0.0,
                    magnetic_field=0.0,
                    periodic=True,
                    spinless=False,
                    particle_hole_symmetry=False,
                    sc_gap=1.0,
                    qsim_option=qsim_option)
                
                gamma, beta = optimize_by_gradient_descent_multiprocess(
                    function=partial(get_expectation_bcs_hubbard, function_args=function_args), 
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
    
if __name__=='__main__':
    main()