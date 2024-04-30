import toml
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# TOMLファイルからディレクトリを読み込む
with open('graphics.toml', 'r') as f:
    config = toml.load(f)
    directory = config['directory']
    csv_prefix = config['csv_prefix']
    save_fig_directory = config['save_fig_directory']

# number_lの範囲を定義
number_l_range = range(8, 17, 2)
# number_pの範囲を定義
number_p_range = range(1, 11)

# マーカーと線種のリストを定義
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'H', '*']
linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']

plt.figure(figsize=(10, 6))  # 一度だけ図のサイズを定義

T_values = {}

# 各number_l, number_pに対して最新のファイルを検索し、最終行のデータを抽出
for i, number_l in enumerate(number_l_range):
    number_p = number_l / 2
    pattern = os.path.join(directory, f"{csv_prefix}_l{number_l:02}_p{int(number_p)}_*.csv")
    files = glob.glob(pattern)
    if files:
        latest_file = max(files, key=os.path.getmtime)
        df = pd.read_csv(latest_file)
        # 'gamma[N]'と'beta[N]'の合計をTとする
        gamma_columns = [col for col in df.columns if 'gamma' in col]
        beta_columns = [col for col in df.columns if 'beta' in col]
        T_values[number_l] = df[gamma_columns + beta_columns].sum(axis=1).iloc[-1]

# y軸のメジャーフォーマッターを整数部分のみに設定
# plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x)}'))

# グラフにプロット
plt.plot(list(T_values.keys()), list(T_values.values()), marker='o')

plt.tick_params(axis='both', labelsize=16)  # x軸とy軸の目盛りラベルのフォントサイズ
plt.xlabel('Number_l', fontsize=20)  # x軸ラベルのフォントサイズ
plt.ylabel('T', fontsize=20)  # y軸ラベルのフォントサイズ
plt.grid(True)
print(list(T_values.keys()))
print(list(T_values.values()))
# plt.yticks(min(list(T_values.values())),max(list(T_values.values()))+1,1)

# 画像ファイルとして保存
plt.savefig(os.path.join(save_fig_directory, f"{csv_prefix}_T_vs_number_l.png"), format='png', dpi=300)
plt.close()  # プロット後にクローズする
