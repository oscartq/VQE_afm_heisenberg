import toml
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

# Read directories from the TOML file
with open('graphics.toml', 'r') as f:
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

# For each number_l and number_p, search for the latest file and extract data from the last row
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
                # Get the value from the 'energy' column in the last row and extract the real part of the complex number
                energy_value = df['energy'].iloc[-1]
                energy_real = np.real(complex(energy_value.replace('j', 'j')))
                energy_per_length_values[number_p] = energy_real

    # Plot the graph with different markers and linestyles for each l_number
    plt.plot(list(energy_per_length_values.keys()), list(energy_per_length_values.values()),
             marker=markers[i % len(markers)], linestyle=linestyles[i % len(linestyles)], label=f'L = {number_l}')

plt.tick_params(axis='both', labelsize=16)  # Font size for x and y axis tick labels
plt.xlabel('$p$', fontsize=20)  # Font size for x-axis label
plt.ylabel('$E$', fontsize=20)  # Font size for y-axis label
plt.title(f"Energy per length vs. p-number\n{csv_prefix}", fontsize=16)  # Font size for the title
plt.legend(fontsize=20)  # Font size for the legend
plt.grid(True)

# Save the plot as an image file
plt.savefig(os.path.join(save_fig_directory, f"{csv_prefix}_energy_vs_p.png"), format='png', dpi=300)
plt.close()  # Close the plot after plotting