import toml
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Read the directory from the TOML file
with open('graphics.toml', 'r') as f:
    config = toml.load(f)
    directory = config['directory']
    csv_prefix = config['csv_prefix']
    save_fig_directory = config['save_fig_directory']
    number_l_list = config['number_l']
    number_p_list = config['number_p']

# Define the range for number_l
# number_l_range = range(8, 19, 2)
# Define the range for number_p
# number_p_range = range(1, 11)

# Define lists of markers and linestyles
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'H', '*']
linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']

plt.figure(figsize=(10, 6))  # Define the size of the figure only once

T_values = {}

# For each number_l and number_p, search for the latest file and extract the data from the last row
for i, number_l in enumerate(number_l_list):
    number_p = int(number_l / 2)
    pattern = os.path.join(directory, f"{csv_prefix}_l{number_l:02}_p{int(number_p)}_*.csv")
    files = glob.glob(pattern)
    if files:
        latest_file = max(files, key=os.path.getmtime)
        df = pd.read_csv(latest_file)
        # Sum of 'gamma[N]' and 'beta[N]' columns is T
        gamma_columns = [col for col in df.columns if 'gamma' in col]
        beta_columns = [col for col in df.columns if 'beta' in col]
        T_values[number_l] = df[gamma_columns + beta_columns].sum(axis=1).iloc[-1]

# Set the major formatter for the y-axis to only show integer parts
# plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x)}'))

# Plot on the graph
plt.plot(list(T_values.keys()), list(T_values.values()), marker='o')

plt.tick_params(axis='both', labelsize=16)  # Font size for tick labels on x-axis and y-axis
plt.xlabel('Number_l', fontsize=20)  # Font size for x-axis label
plt.ylabel('T', fontsize=20)  # Font size for y-axis label
plt.grid(True)
plt.legend(loc='upper right')
print(list(T_values.keys()))
print(list(T_values.values()))
# plt.yticks(min(list(T_values.values())),max(list(T_values.values()))+1,1)

# Save as an image file
plt.savefig(os.path.join(save_fig_directory, f"{csv_prefix}_T_vs_number_l.png"), format='png', dpi=300)
plt.close()  # Close the plot after plotting