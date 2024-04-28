import os
import datetime
from functools import partial
import tomllib

import cirq
import openfermion
import numpy as np
from anzats import Anzats
from expectation import get_expectation_critical_state, TFIMStateArgs
from optimization import optimize_by_gradient_descent


def main():
    with open(".toml", mode="rb") as f:
        config = tomllib.load(f)
        print(config)
    length_list = config["afm_heisenberg"]["length_list"]
    p_list = config["afm_heisenberg"]["p_list"]
    alpha = config["afm_heisenberg"]["alpha"]
    delta_gamma = config["afm_heisenberg"]["delta_gamma"]
    delta_beta  =config["afm_heisenberg"]["delta_beta"]
    iteration = config["afm_heisenberg"]["iteration"]

    results_dir_path = config["afm_heisenberg"]["results_dir_path"]

    if not os.path.exists(results_dir_path):
        os.mkdir(results_dir_path)

if __name__=='__main__':
    main()