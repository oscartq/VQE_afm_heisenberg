import os
import sys
import datetime
from functools import partial
import tomllib
import multiprocessing as mp

import cirq
import openfermion
import numpy as np
import scipy
import scipy.optimize 
from autograd import grad

from anzats import Anzats
from expectation import get_expectation_afm_heisenberg, AFMHeisenbergArgs
from optimization import optimize_by_gradient_descent_multiprocess, optimize_by_gradient_descent


def get_gradient_for_scipy(function, param: np.array, function_args):
    gamma, beta = np.split(param, 2)
    grad_gamma = np.zeros_like(gamma)
    grad_beta  = np.zeros_like(beta)
    gamma_edge = gamma
    beta_edge  = beta
    delta_gamma = 1e-5
    delta_beta  = 1e-5
    # initial gamma, beta?
    
    if not (gamma.size == beta.size):
        return 1

    for index in range(gamma.size):
        center = gamma[index]
        gamma_edge[index] = gamma[index] - delta_gamma
        e1 = function(param=param)
        gamma_edge[index] = gamma[index] + delta_gamma
        e2 = function(param=param)
        grad_gamma[index] = (e2.real-e1.real)/(2*delta_gamma)
        gamma[index] = center

        center = beta[index]
        beta_edge[index] = beta[index] - delta_beta
        e1 = function(param=param)
        beta_edge[index] = beta[index] + delta_beta
        e2 = function(param=param)
        grad_beta[index] = (e2.real-e1.real)/(2*delta_beta)
        beta[index] = center
    
    return np.concatenate([grad_gamma, grad_beta])


def main():
    output_file_prefix = "4-5_afm-heisenberg"

    # input a config file
    with open(os.path.join(os.path.dirname(sys.argv[0]), ".toml"), mode="rb") as f:
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


    # run for p and l
    for p in p_list:
        initial_gamma = np.array([0.5 for i in range(p)])
        initial_beta  = np.array([0.5 for i in range(p)])
        for length in length_list:
            qsim_option = {}
            # qsim_option = {'t': 1, 'f':1}
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

            function_args = AFMHeisenbergArgs(length, qsim_option)


            out = []
            def callback(xk):
                out.append(xk)
                print(xk)

            
            # scipy.optimize.minimize requires one variational parameter
            param = np.concatenate([initial_gamma, initial_beta])

            # distribute param to gamma and beta, set function_args
            def get_expectation_afm_heisenberg_with_concated_parameters(param):
                gamma, beta = np.split(param, 2)
                return get_expectation_afm_heisenberg(function_args=function_args,
                                                        gamma=gamma,
                                                        beta =beta
                                                        ).real
            
            # run with jac="3-point and judge convergence with tol=1e-12 and 'gtol': 1e-12
            minimized = scipy.optimize.minimize(
                fun=get_expectation_afm_heisenberg_with_concated_parameters,
                x0=param,
                jac="3-point",
                method='L-BFGS-B',
                options={'maxiter': iteration, 'gtol': 1e-12},
                bounds=[(0,None)]*len(param),
                tol=1e-12,
                )

            # split parameters, get values
            gamma, beta = np.split(minimized['x'], 2)
            energy = minimized['fun']
            iterated = minimized['nit']
            print(minimized)

    
if __name__=='__main__':
    main()