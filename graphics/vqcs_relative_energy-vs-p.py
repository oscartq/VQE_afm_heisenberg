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
with open('graphics.toml', 'r') as f:
    config = toml.load(f)
    directory = config['directory']
    csv_prefix = config['csv_prefix']
    save_fig_directory = config['save_fig_directory']
    number_l_list = config['number_l']
    number_p_list = config['number_p']

if not os.path.exists(save_fig_directory):
    os.mkdir(save_fig_directory)
    print(f"Directory {save_fig_directory} created.")
if os.path.exists(save_fig_directory):
    shutil.rmtree(save_fig_directory)
    os.mkdir(save_fig_directory)
    print(f"Directory {save_fig_directory} cleared.")
    
# Define lists of markers and linestyles
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'H', '*']
linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']

plt.figure(figsize=(10, 6))  # Define the size of the figure only once

# Search for the latest file for each number_l, number_p, and extract the data from the last row
for i, number_l in enumerate(number_l_list):
    energy_per_length_values = {}
    for number_p in number_p_list:
        pattern = os.path.join(directory, f"{csv_prefix}_l{number_l:02}_p{number_p}_*.csv")
        files = glob.glob(pattern)
        if files:
            latest_file = max(files, key=os.path.getmtime)
            print(latest_file)
            df = pd.read_csv(latest_file)
            if 'energy' in df.columns:
                # Get the value of the energy column from the last row and extract the real part of the complex number
                energy_value = df['energy'].iloc[-1]
                energy_real = energy_value #np.real(complex(energy_value.replace('j', 'j')))
                exact_energy, state = get_exact_expectation_afm_heisenberg(number_l)
                
                energy_per_length_values[number_p] = energy_real/exact_energy

    # Plot the graph for each l_number with different markers and linestyles
    plt.plot(list(energy_per_length_values.keys()), list(energy_per_length_values.values()),
             marker=markers[i % len(markers)], linestyle=linestyles[i % len(linestyles)], label=f'L = {number_l}')

plt.tick_params(axis='both', labelsize=16)  # Font size of the tick labels on the x-axis and y-axis
plt.xlabel('$p$', fontsize=20)  # Font size of the x-axis label
plt.ylabel('$E$/$E_{exact}$', fontsize=20)  # Font size of the y-axis label
plt.title(f"Relative energy vs. p-number\n{csv_prefix}", fontsize=16)  # Font size of the title
plt.legend(fontsize=20)  # Font size of the legend
plt.grid(True)

# Save as an image file
plt.savefig(os.path.join(save_fig_directory, f"{csv_prefix}_relative_energy_vs_p.pdf"), format='pdf', dpi=300)
plt.close()  # Close after plotting