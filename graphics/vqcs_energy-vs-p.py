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
    number_l_list = config['number_l']
    number_p_list = config['number_p']


# マーカーと線種のリストを定義
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'H', '*']
linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']

plt.figure(figsize=(10, 6))  # 一度だけ図のサイズを定義

# 各number_l, number_pに対して最新のファイルを検索し、最終行のデータを抽出
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
                # 最終行のenergy列の値を取得し、複素数の実数部を抽出
                energy_value = df['energy'].iloc[-1]
                energy_real = np.real(complex(energy_value.replace('j', 'j')))
                energy_per_length_values[number_p] = energy_real

    # 各l_numberごとに異なるマーカーと線種でグラフをプロット
    plt.plot(list(energy_per_length_values.keys()), list(energy_per_length_values.values()),
             marker=markers[i % len(markers)], linestyle=linestyles[i % len(linestyles)], label=f'L = {number_l}')

plt.tick_params(axis='both', labelsize=16)  # x軸とy軸の目盛りラベルのフォントサイズ
plt.xlabel('p', fontsize=20)  # x軸ラベルのフォントサイズ
plt.ylabel('Energy', fontsize=20)  # y軸ラベルのフォントサイズ
# plt.title('Energy per length vs. p_number', fontsize=16)  # タイトルのフォントサイズ
plt.legend(fontsize=20)  # 凡例のフォントサイズ
plt.grid(True)

# 画像ファイルとして保存
plt.savefig(os.path.join(save_fig_directory, f"{csv_prefix}_energy_vs_p.png"), format='png', dpi=300)
plt.close()  # プロット後にクローズする
