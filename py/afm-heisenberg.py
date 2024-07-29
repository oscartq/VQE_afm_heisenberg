import os
import shutil
import datetime
import time
from functools import partial
import tomllib
import multiprocessing as mp
import numpy as np
from expectation import get_expectation_afm_heisenberg, AFMHeisenbergArgs
from optimization import optimize_by_gradient_descent_multiprocess, optimize_by_gradient_descent, optimize_by_lbfgsb

def main():  # Main function
    output_file_prefix = "afm-heisenberg"  # Prefix for output files

    # Load configuration from a TOML file
    with open(".toml", mode="rb") as f:
        config = tomllib.load(f)

    # Retrieve parameters from the configuration
    length_list = config[output_file_prefix]["length_list"]
    p_list = config[output_file_prefix]["p_list"]
    alpha = config[output_file_prefix]["alpha"]
    delta_gamma = config[output_file_prefix]["delta_gamma"]
    delta_beta = config[output_file_prefix]["delta_beta"]
    iteration = config[output_file_prefix]["iteration"]
    optimization = config[output_file_prefix]["optimization"]
    boundary_condition = config[output_file_prefix]["boundary_condition"]
    results_dir_path = config[output_file_prefix]["results_dir_path"]    

    # Set boundary condition: Periodic (PBC) or Open (OBC)
    if boundary_condition == "PBC":
        periodic = True
    elif boundary_condition == "OBC":
        periodic = False
    else:
        periodic = False
        print(f'{boundary_condition} not valid boundary condition, using OBC.')
    
    # Create results directory if it doesn't exist, or clear it if it does
    if not os.path.exists(results_dir_path):
        os.mkdir(results_dir_path)
        print(f"Directory {results_dir_path} created.")
    if os.path.exists(results_dir_path):
        shutil.rmtree(results_dir_path)
        os.mkdir(results_dir_path)
        print(f"Directory {results_dir_path} cleared.")

    # Set the timezone to Japan Standard Time (JST)
    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, 'JST')
    now = datetime.datetime.now(JST)
    ymdhms = now.strftime('%Y-%m-%d_%H-%M-%S')  # Current time formatted as a string

    start_time = time.time()  # Start timing the execution

    if optimization == "gradient-descent":  # Check if using gradient descent optimization
        print('Running Gradient Descent optimizer')
        pool = mp.Pool(4)  # Create a pool of 4 parallel processes

        # Loop over values of p and length
        for p in p_list:
            initial_gamma = np.array([0.6 for _ in range(p)])  # Initialize gamma values
            initial_beta = np.array([0.6 for _ in range(p)])  # Initialize beta values

            for length in length_list:
                qsim_option = {'t': int(length / 2), 'f': 1}  # Set simulation options
                csvpath = os.path.join(results_dir_path, '{}_l{:02}_p{}_{}.csv'.format(output_file_prefix, length, p, ymdhms))
                tomlpath = os.path.join(results_dir_path, '{}_l{:02}_p{}_{}.toml'.format(output_file_prefix, length, p, ymdhms))

                # Write parameters to a TOML file
                with open(tomlpath, mode='a') as f:
                    f.write("length       ={}\n".format(length))
                    f.write("p            ={}\n".format(p))
                    f.write("alpha        ={}\n".format(alpha))
                    f.write("initial_gamma={}\n".format("[" + ", ".join(str(value) for value in initial_gamma.tolist()) + "]"))
                    f.write("initial_beta ={}\n".format("[" + ", ".join(str(value) for value in initial_beta.tolist()) + "]"))
                    f.write("delta_gamma  ={}\n".format(delta_gamma))
                    f.write("delta_beta   ={}\n".format(delta_beta))
                    f.write("iteration    ={}\n".format(iteration))

                function_args = AFMHeisenbergArgs(length, periodic, qsim_option)  # Create function arguments

                # Perform optimization using gradient descent
                gamma, beta = optimize_by_gradient_descent_multiprocess(
                    function=partial(get_expectation_afm_heisenberg, function_args=function_args),
                    initial_gamma=initial_gamma,
                    initial_beta=initial_beta,
                    alpha=alpha,
                    delta_gamma=delta_gamma,
                    delta_beta=delta_beta,
                    iteration=iteration,
                    tol=1e-8,
                    figure=True,
                    filepath=csvpath,
                    pool=pool)

        pool.close()  # Close the pool
        pool.join()  # Wait for the pool to finish

    elif optimization == "scipy":  # Check if using Scipy optimization
        print('Running Scipy optimizer')
        for p in p_list:
            initial_gamma = np.array([0.6 for _ in range(p)])  # Initialize gamma values
            initial_beta = np.array([0.6 for _ in range(p)])  # Initialize beta values

            for length in length_list:
                qsim_option = {'t': int(length / 2), 'f': 1}  # Set simulation options
                csvpath = os.path.join(results_dir_path, '{}_l{:02}_p{}_{}.csv'.format(output_file_prefix, length, p, ymdhms))
                tomlpath = os.path.join(results_dir_path, '{}_l{:02}_p{}_{}.toml'.format(output_file_prefix, length, p, ymdhms))

                # Write parameters to a TOML file
                with open(tomlpath, mode='a') as f:
                    f.write("length       ={}\n".format(length))
                    f.write("p            ={}\n".format(p))
                    f.write("initial_gamma={}\n".format("[" + ", ".join(str(value) for value in initial_gamma.tolist()) + "]"))
                    f.write("initial_beta ={}\n".format("[" + ", ".join(str(value) for value in initial_beta.tolist()) + "]"))

                function_args = AFMHeisenbergArgs(length, periodic, qsim_option)  # Create function arguments

                # Perform optimization using Scipy's L-BFGS-B algorithm
                gamma, beta = optimize_by_lbfgsb(
                    function=partial(get_expectation_afm_heisenberg, function_args=function_args),
                    initial_gamma=initial_gamma,
                    initial_beta=initial_beta,
                    bounds=None, #[(0, 1)] * (2 * p),
                    print_results=True,
                    filepath=csvpath)
    else:
        print(f'Error no optimization method named {optimization} available')  # Error message for unknown optimization method

    end_time = time.time()  # End timing the execution
    elapsed_time = end_time - start_time  # Calculate elapsed time

    # Write elapsed time to a file
    elapsed_time_file = os.path.join(results_dir_path, '{}_elapsed_time_{}.txt'.format(output_file_prefix, ymdhms))
    with open(elapsed_time_file, mode='w') as f:
        f.write("Elapsed time: {:.2f} seconds".format(elapsed_time))

if __name__ == '__main__':
    main()  # Execute the main function if the script is run directly