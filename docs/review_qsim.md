## [Get started with qsimcirq  |  Quantum Simulator  |  Google Quantum AI](https://quantumai.google/qsim/tutorials/qsimcirq)
`pip3 install qsimcirq`

基本的には `simulator = cirq.Simulator()` の代わりに `simulator = qsimcirq.QSimSimulator()` を使えば、同じインターフェースで`simulator`を操作できる。`simulator = qsimcirq.QSimSimulator(options)` として設定をすると、並列化や算術強度?を変えて高速化できる。
cirq と qsimcirq の測定では基底の順番が異なるため、同じ測定でも見かけ上のCLIの出力が異なる場合がある

## [Before you begin  |  Quantum Simulator  |  Google Quantum AI](https://quantumai.google/qsim/tutorials/gcp_before_you_begin)
qsim を動かすクラウドコンピュータの設定やコストに関する簡単な案内。

## [CPU-based quantum simulation on Google Cloud  |  Quantum Simulator  |  Google Quantum AI](https://quantumai.google/qsim/tutorials/gcp_cpu)
Google Cloud 上の CPU で実行する場合のガイド。

## [GPU-based quantum simulation on Google Cloud  |  Quantum Simulator  |  Google Quantum AI](https://quantumai.google/qsim/tutorials/gcp_gpu)
### windows11+ubuntu22.04 (wsl)
Option1 を選択した。

`CUDA_ARCHITECTURES is empty ...`
[CUDA GPUs - Compute Capability | NVIDIA Developer](https://developer.nvidia.com/cuda-gpus) を見て小数点を除いた値を`pybind_interface/cuda/CMakeLists.txt`に入力する
    `CUDA_ARCHITECTURES=75`
