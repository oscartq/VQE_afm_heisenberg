import os
import datetime
from functools import partial
import tomllib

import cirq
import openfermion
import numpy as np
from anzats import Anzats
from expectation import get_expectation_critical_state, TFIMStateArgs
from optimization import optimize_by_gradient_descent, optimize_by_gradient_descent_gpu

Pi=3.1415


def optimize_critical_state(length, 
                            g,
                            p, 
                            alpha, 
                            delta_gamma, 
                            delta_beta, 
                            initial_gamma,
                            initial_beta,
                            iteration, 
                            csvpath, 
                            tomlpath,
                            gpu=False):

    function_args = TFIMStateArgs(length, g)

    print("exact energy: ")
    print(-1.28*length)

    if gpu:
        gamma, beta = optimize_by_gradient_descent_gpu(partial(get_expectation_critical_state, function_args=function_args), initial_gamma, initial_beta, alpha, delta_gamma, delta_beta, iteration, True, csvpath)

    else:
        gamma, beta = optimize_by_gradient_descent(partial(get_expectation_critical_state, function_args=function_args), initial_gamma, initial_beta, alpha, delta_gamma, delta_beta, iteration, True, csvpath)
    print(gamma, beta)


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

def main():
    with open(".toml", mode="rb") as f:
        config = tomllib.load(f)
        print(config)
    length_list = config["critical_state"]["length_list"]
    p_list = config["critical_state"]["p_list"]
    g = config["critical_state"]["g"]
    alpha = config["critical_state"]["alpha"]
    delta_gamma = config["critical_state"]["delta_gamma"]
    delta_beta  = config["critical_state"]["delta_beta"]
    initial_gamma = np.array(config["critical_state"]["initial_gamma"])
    initial_beta  = np.array(config["critical_state"]["initial_beta"])
    iteration = config["critical_state"]["iteration"]
    gpu = config["critical_state"]["gpu"]

    if not (len(initial_gamma)==p_list[0]):
        raise ValueError("length of the initial parameter gamma must equal to p")
    if not (len(initial_beta)==p_list[0]):
        raise ValueError("length of the initial parameter beta must equal to p")
        
    results_dir_path = config["critical_state"]["results_dir_path"]
    # results_dir_path = os.path.join('.results')
    if not os.path.exists(results_dir_path):
        os.mkdir(results_dir_path)
    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, 'JST')
    now = datetime.datetime.now(JST)
    ymdhms = now.strftime('%Y-%m-%d_%H-%M-%S')


    for p in p_list:
        if not (len(initial_gamma)==p):
            initial_gamma = np.array([0.5 for i in range(p)])
            initial_beta  = np.array([0.5 for i in range(p)])
        for length in length_list:
            csvpath = os.path.join(results_dir_path, '4-2_critical_l{:02}_p{}_{}.csv'.format(length, p, ymdhms))
            tomlpath = os.path.join(results_dir_path, '4-2_critical_l{:02}_p{}_{}.toml'.format(length, p, ymdhms))
            optimize_critical_state(length, 
                            g,
                            p, 
                            alpha, 
                            delta_gamma, 
                            delta_beta, 
                            initial_gamma,
                            initial_beta,
                            iteration, 
                            csvpath, 
                            tomlpath,
                            gpu)

if __name__=='__main__':
    main()