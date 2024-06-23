import os
import sys
import datetime
from functools import partial
import tomllib
import multiprocessing as mp

import cirq
import openfermion
import numpy as np
from anzats import Anzats, AnzatsAFMHeisenberg
from expectation import get_expectation_afm_heisenberg, AFMHeisenbergArgs
from optimization import optimize_by_gradient_descent_multiprocess, optimize_by_gradient_descent

import sympy 
Pi=3.1415


def main():
    output_file_prefix = "4-5_afm-heisenberg"

    # input a config file
    with open(".toml", mode="rb") as f:
        config = tomllib.load(f)

    length_list = config[output_file_prefix]["length_list"]
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

    pool = mp.Pool(1)

    # run for p and l
    for p in p_list:
        initial_gamma = np.array([Pi/2 * sympy.Symbol(f"\\gamma_{i}") for i in range(p)])
        initial_beta  = np.array([Pi/2 * sympy.Symbol(f"\\beta_{i}") for i in range(p)])


        for length in length_list:
            anzats = AnzatsAFMHeisenberg(length, initial_gamma, initial_beta)

            texpath = os.path.join(os.path.dirname(sys.argv[0]), '.results_texts', f'latex_qcircuit_l{length:02}_p{p:02}.tex')
            if not os.path.exists(os.path.dirname(texpath)):
                os.mkdir(os.path.dirname(texpath))

            # output and replace latex qcircuit text
            with open(texpath, mode='w') as f:
                tex = anzats.circuit_to_latex_using_qcircuit()
                tex = tex.replace("-1.0*", '')
                f.write(tex)
                print(tex)


    
if __name__=='__main__':
    main()