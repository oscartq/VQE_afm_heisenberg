import toml
import glob
import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import sys

# Add the ../py/ directory to the system path
script_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.join(script_dir, '../py')
sys.path.insert(0, module_dir)

from exact_expectation import get_exact_expectation_afm_heisenberg

# Read the directory from the TOML file
with open('graphics_error.toml', 'r') as f:
    config = toml.load(f)
    directory_list = config['directory']
    csv_prefix = config['csv_prefix']
    save_fig_directory = config['save_fig_directory']
    number_l_list = config['number_l']
    number_p_list = config['number_p']

if not os.path.exists(save_fig_directory):
    os.mkdir(save_fig_directory)
    print(f"Directory {save_fig_directory} created.")
else:
    shutil.rmtree(save_fig_directory)
    os.mkdir(save_fig_directory)
    print(f"Directory {save_fig_directory} cleared.")

# Define lists of markers and linestyles
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'H', '*']
linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']

# Prepare data for plotting
plot_data = {}

for dir in directory_list:
    print(f"Now working on {dir}")
    dir_name = os.path.basename(dir)
    for number_l in number_l_list:
        if number_l not in plot_data:
            plot_data[number_l] = {}
        energy_error_per_length_values = {}
        for number_p in number_p_list:
            pattern = os.path.join(dir, f"{csv_prefix}_l{number_l:02}_p{number_p}_*.csv")
            files = glob.glob(pattern)
            if files:
                latest_file = max(files, key=os.path.getmtime)
                print(latest_file)
                df = pd.read_csv(latest_file)
                if 'energy' in df.columns:
                    # Get the value of the energy column from the last row
                    energy_value = df['energy'].iloc[-1]
                    exact_energy, state = get_exact_expectation_afm_heisenberg(number_l)
                    energy_error_per_length_values[number_p] = (exact_energy - energy_value) / exact_energy

        # Collect data for plotting
        plot_data[number_l][dir_name] = energy_error_per_length_values

# Plotting each figure for each L value
for number_l, dir_data in plot_data.items():
    plt.figure(figsize=(10, 6))  # Define the size of the figure only once
    for i, (dir_name, energy_error_per_length_values) in enumerate(dir_data.items()):
        plt.plot(list(energy_error_per_length_values.keys()), list(energy_error_per_length_values.values()),
                 marker=markers[i % len(markers)], linestyle=linestyles[i % len(linestyles)], label=dir_name)
    
    plt.tick_params(axis='both', labelsize=16)  # Font size of the tick labels on the x-axis and y-axis
    plt.xlabel('$p$', fontsize=20)  # Font size of the x-axis label
    plt.ylabel('$E_{error}$', fontsize=20)  # Font size of the y-axis label
    plt.title(f"Relative energy error vs. p-number for L = {number_l}\n{csv_prefix}", fontsize=16)  # Font size of the title
    plt.legend(fontsize=20)  # Font size of the legend
    plt.grid(True)
    
    # Save as an image file
    plt.savefig(os.path.join(save_fig_directory, f"{csv_prefix}_L{number_l:02}_error_optimization.pdf"), format='pdf', dpi=300)
    plt.close()  # Close after plotting
