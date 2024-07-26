import os
import shutil
import datetime
import time
from functools import partial
import tomllib
import numpy as np  
from expectation import get_expectation_afm_heisenberg_matrix, AFMHeisenbergMatrixArgs
from optimization import optimize_by_lbfgsb

def main():  # Main function
    output_file_prefix = "afm-heisenberg-matrix"  # Prefix for output files
    
    # Load configuration from a TOML file
    with open(".toml", mode="rb") as f:
        config = tomllib.load(f)

    # Retrieve parameters from the configuration
    rows_list = config[output_file_prefix]["rows_list"]
    cols_list = config[output_file_prefix]["cols_list"]
    p_list = config[output_file_prefix]["p_list"]
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
    else:
        shutil.rmtree(results_dir_path)
        os.mkdir(results_dir_path)
        print(f"Directory {results_dir_path} cleared.")

    # Set the timezone to Japan Standard Time (JST)
    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, 'JST')
    now = datetime.datetime.now(JST)
    ymdhms = now.strftime('%Y-%m-%d_%H-%M-%S')  # Current time formatted as a string
    
    start_time = time.time()  # Start timing the execution
    
    print('Running Scipy optimizer')
    # Loop over values of p
    for p in p_list:
        # Initialize parameters
        initial_gamma = np.array([0.6 for _ in range(p)])
        initial_beta = np.array([0.6 for _ in range(p)])
        initial_phi = np.array([0.6 for _ in range(p)])
        initial_theta = np.array([0.6 for _ in range(p)])
        
        rows = rows_list[0]
        cols = cols_list[0]
        length = rows * cols  # Calculate total length
        
        qsim_option = {'t': int(length / 2), 'f': 1}  # Set simulation options
        csvpath = os.path.join(results_dir_path, '{}_l{:02}_p{}_{}.csv'.format(output_file_prefix, length, p, ymdhms))
        tomlpath = os.path.join(results_dir_path, '{}_l{:02}_p{}_{}.toml'.format(output_file_prefix, length, p, ymdhms))

        # Write parameters to a TOML file
        with open(tomlpath, mode='a') as f:
            f.write("length       ={}x{}\n".format(rows, cols))
            f.write("p            ={}\n".format(p))
            f.write("initial_gamma={}\n".format("[" + ", ".join(str(value) for value in initial_gamma.tolist()) + "]"))
            f.write("initial_beta ={}\n".format("[" + ", ".join(str(value) for value in initial_beta.tolist()) + "]"))
            f.write("initial_phi  ={}\n".format("[" + ", ".join(str(value) for value in initial_phi.tolist()) + "]"))
            f.write("initial_theta={}\n".format("[" + ", ".join(str(value) for value in initial_theta.tolist()) + "]"))
        
        # Create function arguments
        function_args = AFMHeisenbergMatrixArgs(rows, cols, periodic, qsim_option)

        # Perform optimization using Scipy's L-BFGS-B algorithm
        gamma, beta, phi, theta = optimize_by_lbfgsb(
            function=partial(get_expectation_afm_heisenberg_matrix, function_args=function_args),
            initial_gamma=initial_gamma,
            initial_beta=initial_beta,
            initial_phi=initial_phi,
            initial_theta=initial_theta,
            bounds=None,
            parameters=4,
            print_results=True,
            filepath=csvpath)
                
    end_time = time.time()  # End timing the execution
    elapsed_time = end_time - start_time  # Calculate elapsed time

    # Write elapsed time to a file
    elapsed_time_file = os.path.join(results_dir_path, '{}_elapsed_time_{}.txt'.format(output_file_prefix, ymdhms))
    with open(elapsed_time_file, mode='w') as f:
        f.write("Elapsed time: {:.2f} seconds".format(elapsed_time))

if __name__ == '__main__':
    main()  # Execute the main function if the script is run directly
