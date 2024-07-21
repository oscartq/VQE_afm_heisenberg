import os
import sys
import toml
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from exact_expectation import get_exact_expectation_afm_heisenberg_lattice

# Read directories from the TOML file
with open(os.path.join(os.path.dirname(sys.argv[0]), 'graphics.toml'), 'r') as f:
    config = toml.load(f)
    directory = config['directory']
    csv_prefix = config['csv_prefix']
    save_fig_directory = config['save_fig_directory']
    number_l_list = config['number_l']
    number_p_list = config['number_p']

# Define lists for markers and linestyles
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'H', '*']
linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']

plt.figure(figsize=(10, 6))  # Define the figure size once

# For each number_l, number_p, search for the latest file and extract the energy data over iterations
for i, number_l in enumerate(number_l_list):
    for number_p in number_p_list:
        pattern = os.path.join(directory, f"{csv_prefix}_l{number_l:02}_p{number_p}_*.csv")
        files = glob.glob(pattern)
        if files:
            latest_file = max(files, key=os.path.getmtime)
            print(latest_file)
            df = pd.read_csv(latest_file)
            if 'energy' in df.columns:
                iterations = df['iter']
                energy_values = df['energy']
                energy_real_values = [np.real(complex(energy.replace('j', 'j'))) for energy in energy_values]
                
                # Plot energy vs. iteration for each file
                plt.plot(iterations, energy_real_values, 
                         marker=markers[i % len(markers)], linestyle=linestyles[i % len(linestyles)], 
                         label=f'L = {number_l}, p = {number_p}')

# Add the exact solution line
for number_l in number_l_list:
    exact_energy, _ = get_exact_expectation_afm_heisenberg_lattice(int(number_l / 2), 2)
    plt.axhline(y=exact_energy, color='r', linestyle='--', label=f'Exact solution L = {number_l}')

plt.tick_params(axis='both', labelsize=16)  # Font size for x and y axis labels
plt.xlabel('Iteration', fontsize=20)  # Font size for x-axis label
plt.ylabel('Energy', fontsize=20)  # Font size for y-axis label
plt.legend(fontsize=10)  # Font size for legend
plt.legend(loc='upper right')
plt.grid(True)

# Save the plot as an image file
plt.savefig(os.path.join(save_fig_directory, f"{csv_prefix}_energy_iteration.pdf"), format='pdf', dpi=300)
plt.close()  # Close the plot after saving