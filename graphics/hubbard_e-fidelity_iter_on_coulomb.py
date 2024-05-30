import tomllib
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read directories from the TOML file
with open('graphics.toml', 'rb') as f:
    config = tomllib.load(f)

table_name = '70_bcs_hubbard'
directory = config[table_name]['directory']
csv_prefix = config[table_name]['csv_prefix']
save_fig_directory = config[table_name]['save_fig_directory']
coulomb_list = config[table_name]['coulomb']
number_p_list = config[table_name]['number_p']
number_l = config[table_name]['number_l']
y_top = config[table_name]['y_top']
y_bottom = config[table_name]['y_bottom']

# Define lists for markers and linestyles
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'H', '*']
linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']

# Plot labels and configurations
plot_label_before_equal = "U"
x_label = "iteration"
y_label = "$\\frac{E-E_{ext}}{E_{ext}}$"
title = ""
fontsize = 24
labelsize = 20
markersize = 2
figsize = (10, 6)

# For each number_p and length, search for the latest file and plot the energy fidelity versus iteration
for number_p in number_p_list:
    for length in number_l:
        fig_name = f"e-fidelity_iter/{csv_prefix}_l{length:02}_p{number_p}_e-fidelity_iter_on_coulomb.png"
        plt.figure(figsize=figsize)
        
        for i, coulomb in enumerate(coulomb_list):
            exact_energy = config[table_name][f"exact_energy_l{length:02}Ut{coulomb:02}"]['exact_value']
            pattern = os.path.join(directory, f"{csv_prefix}_Ut{coulomb:02}_l{length:02}_p{number_p}_*.csv")
            files = glob.glob(pattern)
            if files:
                latest_file = max(files, key=os.path.getmtime)
                print(latest_file)
                df = pd.read_csv(latest_file)
                
                # Plot the data
                plt.plot(df['iter'], df['energy'].apply(lambda x: abs((complex(x).real / exact_energy) - 1)),
                         marker=markers[i % len(markers)], linestyle=linestyles[i % len(linestyles)],
                         label=f'{plot_label_before_equal} = {coulomb}', markersize=markersize)

        # Set plot scales and limits
        plt.yscale('log')
        plt.ylim(top=y_top, bottom=y_bottom)
        plt.tick_params(axis='both', labelsize=labelsize)
        plt.xlabel(x_label, fontsize=fontsize)
        plt.ylabel(y_label, fontsize=fontsize)
        plt.title(title, fontsize=fontsize)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=fontsize)
        plt.grid(True)
        plt.tight_layout()

        # Save the plot as an image file
        plt.savefig(os.path.join(save_fig_directory, fig_name), format='png', dpi=150)
        plt.close()  # Close the plot after plotting
