import toml
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

# TOMLファイルからディレクトリを読み込む
with open('graphics.toml', 'r') as f:
    config = toml.load(f)
    directory = config['directory']
    csv_prefix = config['csv_prefix']
    save_fig_directory = config['save_fig_directory']
    coulomb_list = config['coulomb']
    number_p_list = config['number_p']
    length = config['length']

# マーカーと線種のリストを定義
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'H', '*']
linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
plot_label_before_equal = "coulomb" 
x_label = "iteration"
y_label = "energy"
title = ""
fontsize = 20
labelsize = 16
figsize=(10, 6)
# 各coulomb, number_pに対して最新のファイルを検索し、最終行のデータを抽出
for number_p in number_p_list:
    fig_name = f"energy_iter/{csv_prefix}_l{length:02}_p{number_p}_energy_iter_on_coulomb.png"
    plt.figure(figsize=figsize) 
    for i, coulomb in enumerate(coulomb_list):
        energy_values = {}
        pattern = os.path.join(directory, f"{csv_prefix}_Ut{coulomb:02}_l{length:02}_p{number_p}_*.csv")
        files = glob.glob(pattern)
        if files:
            latest_file = max(files, key=os.path.getmtime)
            print(latest_file)
            df = pd.read_csv(latest_file)   
            plt.plot(df['iter'], df['energy'].apply(lambda x: complex(x).real), 
                    marker=markers[i % len(markers)], linestyle=linestyles[i % len(linestyles)], label=f'{plot_label_before_equal} = {coulomb}')

            # if 'energy' in df.columns:
            #     # 最終行のenergy列の値を取得し、複素数の実数部を抽出

            #     energy_value = df['energy'].iloc[-1]
            #     energy_real = np.real(complex(energy_value.replace('j', 'j')))
            #     energy_values[number_p] = energy_real

            # plt.plot(list(energy_values.keys()), list(energy_values.values()),
                    # marker=markers[i % len(markers)], linestyle=linestyles[i % len(linestyles)], label=f'{plot_label_before_equal} = {coulomb}')

    plt.tick_params(axis='both', labelsize=labelsize) 
    plt.xlabel(x_label, fontsize=fontsize) 
    plt.ylabel(y_label, fontsize=fontsize) 
    plt.title(title, fontsize=fontsize) 
    plt.legend(fontsize=fontsize) 
    plt.legend(loc='upper right')
    plt.grid(True)

    # 画像ファイルとして保存
    plt.savefig(os.path.join(save_fig_directory, fig_name), format='png', dpi=300)
    plt.close()  # プロット後にクローズする
