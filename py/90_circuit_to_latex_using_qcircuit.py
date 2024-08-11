import os
import sys
import datetime
import tomllib
import multiprocessing as mp
import numpy as np
from anzats import AnzatsAFMHeisenberg
import sympy 
Pi=np.pi

def main():
    output_file_prefix = "afm-heisenberg"  # Prefix for output files

    # Load configuration from a TOML file
    with open(".toml", mode="rb") as f:
        config = tomllib.load(f)

    # Retrieve configuration parameters
    length_list = config[output_file_prefix]["length_list"]
    p_list = config[output_file_prefix]["p_list"]
    results_dir_path = config[output_file_prefix]["results_dir_path"]
    
    # Create results directory if it doesn't exist
    if not os.path.exists(results_dir_path):
        os.mkdir(results_dir_path)

    # Set timezone to Japan Standard Time (JST)
    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, 'JST')
    now = datetime.datetime.now(JST)
    ymdhms = now.strftime('%Y-%m-%d_%H-%M-%S')  # Format current time as a string

    # Initialize a multiprocessing pool with 1 worker
    pool = mp.Pool(1)

    # Loop over values of p
    for p in p_list:
        # Create symbolic initial parameters for gamma and beta
        initial_gamma = np.array([Pi/2 * sympy.Symbol(f"\\gamma_{i}") for i in range(p)])
        initial_beta  = np.array([Pi/2 * sympy.Symbol(f"\\beta_{i}") for i in range(p)])

        # Loop over values of length
        for length in length_list:
            # Create an instance of the AnzatsAFMHeisenberg class
            anzats = AnzatsAFMHeisenberg(length, initial_gamma, initial_beta)

            # Define the path for the LaTeX output file
            texpath = os.path.join(os.path.dirname(sys.argv[0]), '.results_texts', f'latex_qcircuit_l{length:02}_p{p:02}.tex')
            
            # Create directory for LaTeX file if it doesn't exist
            if not os.path.exists(os.path.dirname(texpath)):
                os.mkdir(os.path.dirname(texpath))

            # Generate and write LaTeX circuit representation
            with open(texpath, mode='w') as f:
                tex = anzats.circuit_to_latex_using_qcircuit()  # Generate LaTeX code
                tex = tex.replace("-1.0*", '')  # Clean up LaTeX code
                f.write(tex)  # Write LaTeX code to file
                print(tex)  # Print LaTeX code to console

if __name__ == '__main__':
    main()  # Execute the main function if the script is run directly
