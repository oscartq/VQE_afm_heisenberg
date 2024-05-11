import tomllib
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

# TOMLファイルからディレクトリを読み込む
with open('graphics.toml', 'rb') as f:
    config = tomllib.load(f)

table_name = '70_bcs_hubbard'
directory = config[table_name]['directory']
csv_prefix = config[table_name]['csv_prefix']
save_fig_directory = config[table_name]['save_fig_directory']
coulomb_list = config[table_name]['coulomb']
number_p_list = config[table_name]['number_p']
number_l = config[table_name]['number_l']

# マーカーと線種のリストを定義
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'H', '*']
linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
plot_label_before_equal = "coulomb" 
x_label = "iteration"
y_label = r"$E-E_{ext}/E_{ext}$"
title = ""
fontsize = 24
labelsize = 20
markersize = 3
figsize=(10, 6)
# 各coulomb, number_pに対して最新のファイルを検索し、最終行のデータを抽出
for number_p in number_p_list:
    for length in number_l:
        fig_name = f"e-fidelity_iter/{csv_prefix}_l{length:02}_p{number_p}_e-fidelity_iter_on_coulomb.png"
        plt.figure(figsize=figsize)
        for i, coulomb in enumerate(coulomb_list):
            exact_energy = config[f"{table_name}"][f"exact_energy_l{length:02}Ut{coulomb:02}"]['exact_value']
            pattern = os.path.join(directory, f"{csv_prefix}_Ut{coulomb:02}_l{length:02}_p{number_p}_*.csv")
            files = glob.glob(pattern)
            if files:
                latest_file = max(files, key=os.path.getmtime)
                print(latest_file)
                df = pd.read_csv(latest_file)   
                plt.plot(df['iter'], df['energy'].apply(lambda x: abs((complex(x).real/exact_energy)-1)), 
                        marker=markers[i % len(markers)], 
                        linestyle=linestyles[i % len(linestyles)], 
                        label=f'{plot_label_before_equal} = {coulomb}',
                        markersize=markersize)

        plt.yscale('log')
        # plt.ylim(top=1.0)
        plt.tick_params(axis='both', labelsize=labelsize) 
        plt.xlabel(x_label, fontsize=fontsize) 
        plt.ylabel(y_label, fontsize=fontsize) 
        plt.title(title, fontsize=fontsize) 
        plt.legend(fontsize=fontsize) 
        plt.legend(loc='right')
        plt.grid(True)

        # 画像ファイルとして保存
        plt.savefig(os.path.join(save_fig_directory, fig_name), format='png', dpi=300)
        plt.close()  # プロット後にクローズする
