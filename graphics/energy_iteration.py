import os
import sys
import toml
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

# For each number_l, number_p, search for the latest file and extract the last row's data
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
                # Extract the real part of the last row in the energy column
                energy_value = df['energy'].iloc[-1]
                energy_real = np.real(complex(energy_value.replace('j', 'j')))
                energy_per_length_values[number_p] = energy_real / number_l

    # Plot each number_l with different markers and linestyles
    if energy_per_length_values:
        plt.plot(list(energy_per_length_values.keys()), list(energy_per_length_values.values()),
                 marker=markers[i % len(markers)], linestyle=linestyles[i % len(linestyles)], label=f'L = {number_l}')

plt.tick_params(axis='both', labelsize=16)  # Font size for x and y axis labels
plt.xlabel('p', fontsize=20)  # Font size for x-axis label
plt.ylabel('Energy / L', fontsize=20)  # Font size for y-axis label
plt.legend(fontsize=20)  # Font size for legend
plt.legend(loc='upper right')
plt.grid(True)

# Save the plot as an image file
plt.savefig(os.path.join(save_fig_directory, f"{csv_prefix}_energy_iteration.png"), format='png', dpi=300)
plt.close()  # Close the plot after saving