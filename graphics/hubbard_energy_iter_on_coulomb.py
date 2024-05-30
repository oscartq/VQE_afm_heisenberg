import toml
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read directories from the TOML file
with open('graphics.toml', 'r') as f:
    config = toml.load(f)
    directory = config['directory']
    csv_prefix = config['csv_prefix']
    save_fig_directory = config['save_fig_directory']
    coulomb_list = config['coulomb']
    number_p_list = config['number_p']
    number_l = config['number_l']

# Define lists for markers and linestyles
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'H', '*']
linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']

# Plot labels and configurations
plot_label_before_equal = "coulomb"
x_label = "iteration"
y_label = "energy"
title = ""
fontsize = 20
labelsize = 16
figsize = (10, 6)

# For each number_p and length, search for the latest file and plot the energy versus iteration
for number_p in number_p_list:
    for length in number_l:
        fig_name = f"energy_iter/{csv_prefix}_l{length:02}_p{number_p}_energy_iter_on_coulomb.png"
        plt.figure(figsize=figsize)
        
        for i, coulomb in enumerate(coulomb_list):
            pattern = os.path.join(directory, f"{csv_prefix}_Ut{coulomb:02}_l{length:02}_p{number_p}_*.csv")
            files = glob.glob(pattern)
            if files:
                latest_file = max(files, key=os.path.getmtime)
                print(latest_file)
                df = pd.read_csv(latest_file)
                
                # Plot the data
                plt.plot(df['iter'], df['energy'].apply(lambda x: complex(x).real),
                         marker=markers[i % len(markers)], linestyle=linestyles[i % len(linestyles)],
                         label=f'{plot_label_before_equal} = {coulomb}')

        # Configure the plot
        plt.tick_params(axis='both', labelsize=labelsize)
        plt.xlabel(x_label, fontsize=fontsize)
        plt.ylabel(y_label, fontsize=fontsize)
        plt.title(title, fontsize=fontsize)
        plt.legend(fontsize=fontsize, loc='upper right')
        plt.grid(True)

        # Save the plot as an image file
        plt.savefig(os.path.join(save_fig_directory, fig_name), format='png', dpi=300)
        plt.close()  # Close the plot after plotting
