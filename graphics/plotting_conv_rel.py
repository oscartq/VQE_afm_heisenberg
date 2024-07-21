import os
import sys
import toml
import glob
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Add the ../py/ directory to the system path
script_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.join(script_dir, '../py')
sys.path.insert(0, module_dir)

from exact_expectation import get_exact_expectation_afm_heisenberg_lattice

# Read the directory from the TOML file
with open(os.path.join(os.path.dirname(sys.argv[0]), 'graphics.toml'), 'r') as f:
    config = toml.load(f)
    directory = config['directory']
    csv_prefix = config['csv_prefix']
    save_fig_directory = config['save_fig_directory']
    number_l_list = config['number_l']
    number_p_list = config['number_p']
    width = config.get("width", None)
    periodic = config['periodic']

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

# Prepare data for plotting
plot_data = []

# For each number_l, number_p, search for the latest file and extract the energy data over iterations
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
                # Get the value of the energy column from the last row
                energy_value = df['energy'].iloc[-1]
                exact_energy, _ = get_exact_expectation_afm_heisenberg_lattice(int(number_l / width), width, periodic)
                energy_per_length_values[number_p] = energy_value / exact_energy
    # Collect data for plotting
    plot_data.append((number_l, energy_per_length_values))

# Plot the first figure: Relative energy vs. p-number
plt.figure(figsize=(10, 6))

for i, (number_l, energy_per_length_values) in enumerate(plot_data):
    plt.plot(list(energy_per_length_values.keys()), list(energy_per_length_values.values()),
             marker=markers[i % len(markers)], linestyle=linestyles[i % len(linestyles)], label=f'L = {number_l}')

plt.tick_params(axis='both', labelsize=16)
plt.xlabel('$p$', fontsize=20)
plt.ylabel('$E$/$E_{exact}$', fontsize=20)
plt.title(f"Relative energy vs. p-number\n{csv_prefix}", fontsize=16)
plt.legend(fontsize=10)
plt.grid(True)

# Save as an image file
plt.savefig(os.path.join(save_fig_directory, f"{csv_prefix}_relative_energy_vs_p.pdf"), format='pdf', dpi=300)
plt.close()

# Plot the second figure: Relative energy vs. p-number with limited y-axis
plt.figure(figsize=(10, 6))

for i, (number_l, energy_per_length_values) in enumerate(plot_data):
    plt.plot(list(energy_per_length_values.keys()), list(energy_per_length_values.values()),
             marker=markers[i % len(markers)], linestyle=linestyles[i % len(linestyles)], label=f'L = {number_l}')

plt.tick_params(axis='both', labelsize=16)
plt.xlabel('$p$', fontsize=20)
plt.ylabel('$E$/$E_{exact}$', fontsize=20)
plt.ylim(0.9, 1.0)
plt.title(f"Relative energy vs. p-number\n{csv_prefix}, limited y-axis", fontsize=16)
plt.legend(fontsize=10)
plt.grid(True)

# Save as an image file
plt.savefig(os.path.join(save_fig_directory, f"{csv_prefix}_relative_energy_vs_p_zoom.pdf"), format='pdf', dpi=300)
plt.close()

# Plot energy convergence over iterations for each number_l and number_p
plt.figure(figsize=(10, 6))

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
                energy_real_values = df['energy']
                
                plt.plot(iterations, energy_real_values, 
                         marker=markers[i % len(markers)], linestyle=linestyles[i % len(linestyles)], 
                         label=f'L = {number_l}, p = {number_p}')

# Add the exact solution line
for number_l in number_l_list:
    exact_energy, _ = get_exact_expectation_afm_heisenberg_lattice(int(number_l / 2), 2, periodic=True)
    plt.axhline(y=exact_energy, color='r', linestyle='--', label=f'Exact solution L = {number_l}')

plt.tick_params(axis='both', labelsize=16)
plt.xlabel('Iteration', fontsize=20)
plt.ylabel('Energy', fontsize=20)
plt.title('Energy convergence \n 2x4 Heisenberg model')
plt.legend(fontsize=10)
plt.grid(True)

# Save as an image file
plt.savefig(os.path.join(save_fig_directory, f"{csv_prefix}_energy_iteration.pdf"), format='pdf', dpi=300)
plt.close()