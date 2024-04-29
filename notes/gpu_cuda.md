[WSL 2 で NVIDIA CUDA を有効にする | Microsoft Learn](https://learn.microsoft.com/ja-jp/windows/ai/directml/gpu-cuda-in-wsl)

- Powershell 上で `nvidia-smi` 
    - `CUDA Version: 12.0`みたいな表示を確認する
    - 今後この版のCUDAとToolKitをインストールする

[WSL2 上の Ubuntu での NVIDIA CUDA ツールキット, NVIDIA cuDNN, PyTorch, TensorFlow 2.11 のインストールと動作確認（Windows 上）](https://www.kkaneko.jp/tools/wsl/wsl_tensorflow2.html)
- `sudo apt -y install cuda-11-8` の代わりに `sudo apt-get -y install cuda-toolkit-12-0` を実行する

- cuda の PATH を通す
    - `ls /usr/local`
    - `cuda-12.x` ディレクトリを探す。その名前を`[DIR]`とする。
    - `echo "export PATH=/usr/local/[DIR]/bin${PATH:+:${PATH} }" >> ~/.bashrc`
    - `source ~/.bashrc`
        - [GPU-based quantum simulation on Google Cloud  |  Quantum Simulator  |  Google Quantum AI](https://quantumai.google/qsim/tutorials/gcp_gpu)