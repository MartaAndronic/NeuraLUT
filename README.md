# NeuraLUT: Hiding Neural Network Density in Boolean Synthesizable Functions

[![DOI](https://img.shields.io/badge/DOI-10.1109/FPL64840.2024.00028-orange)](https://doi.org/10.1109/FPL64840.2024.00028)
[![arXiv](https://img.shields.io/badge/arXiv-2403.00849-b31b1b.svg?style=flat)](https://arxiv.org/abs/2403.00849)

<p align="left">
  <img src="logo.png" width="500" alt="NeuraLUT Logo">
</p>

NeuraLUT is the first quantized neural network training methodology that maps dense and full-precision sub-networks with skip-connections to LUTs to leverage the underlying structure of the FPGA architecture.
> _Built on top of [LogicNets](https://github.com/Xilinx/logicnets), NeuraLUT introduces new architecture designs, optimized training flows, and innovative sparsity handling._
---

#### ✨ New! ReducedLUT branch available for advanced compression using don't-cares (see below).

---

## 🚀 Features

- 🔧 **Quantized training** with sub-networks synthesized into truth tables.
- ⚡️ **Skip connections within LUTs** for better gradient flow and performance.
- 🎯 **Easy FPGA integration** using Vivado and Verilator.
- 📊 **Experiment tracking** with [Weights & Biases](https://wandb.ai/).
- 🧠 Supports **MNIST** and **Jet Substructure Tagging**.
- 🧪 Integration with [Brevitas](https://github.com/Xilinx/brevitas) for quantization-aware training.

---

## 🛠️ Quickstart Guide

### 1. Set Up Conda Environment

> Requires [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

```bash
conda create -n neuralut python=3.12.4
conda activate neuralut
pip install torch==2.4.0 torchvision==0.19.0
```

👉 For CUDA-specific instructions, refer to the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

### 2. Install Brevitas

```bash
conda install -y packaging pyparsing
conda install -y docrep -c conda-forge
pip install --no-cache-dir git+https://github.com/Xilinx/brevitas.git@67be9b58c1c63d3923cac430ade2552d0db67ba5
```

### 3. Install Project Dependencies

```bash
pip install -r requirements.txt
cd NeuraLUT
pip install .
```

### 4. Enable Experiment Tracking

```bash
pip install wandb
wandb.login()
```

---

## 🔧 Optional Tools (for Hardware Integration)

### ✅ Vivado Design Suite

Download and install from [Xilinx Vivado](https://www.xilinx.com/products/design-tools/vivado.html).  
📌 _Used version in our experiments: **Vivado 2020.1**._

### ✅ Verilator

```bash
nix-store --realise /nix/store/q12yxbndfwibfs5jbqwcl83xsa5b0dh8-verilator-4.110
```

### ✅ oh-my-xilinx

```bash
git clone https://github.com/ollycassidy13/oh-my-xilinx.git /path/to/local/dir
export OHMYXILINX=/path/to/local/dir
```

---

## 🌿 ReducedLUT

We released a dedicated [ReducedLUT branch](https://github.com/MartaAndronic/NeuraLUT/tree/reducedlut) which demonstrates the **L-LUT compression pipeline** described in our ReducedLUT paper. This includes:
 
📄 [arXiv](https://arxiv.org/abs/2412.18579) | 📘 [ACM DL](https://dl.acm.org/doi/10.1145/3706628.3708823) | 📦 [Zenodo](https://doi.org/10.5281/zenodo.14499541)

---

## 🧬 What's New in NeuraLUT vs LogicNets?

| Feature | LogicNets | NeuraLUT |
|--------|-----------|-----------|
| **Dataset Support** | Jet Substructure | Jet Substructure, MNIST |
| **Training Flow** | Weight mask for sparsity | FeatureMask for input channel control |
| **Forward Function** | Basic FC layers | Multiple FCs + Skip Connections |
| **Experiment Logging** | TensorBoard | Weights & Biases |
| **GPU Integration** | ✘ | ✅ |
| **Neuron Enumeration** | Basic LUT inference | Batched truth table gen |
| **Architecture Customization** | Limited | Novel model designs described in paper |

---

## 📚 Citation

#### If this repo contributes to your research or FPGA design, please cite our NeuraLUT paper:

```bibtex
@inproceedings{andronic2024neuralut,
	author	= "Andronic, Marta and Constantinides, George A.",
	title		= "{NeuraLUT: Hiding Neural Network Density in Boolean Synthesizable Functions}",
	booktitle	= "{2024 34th International Conference on Field-Programmable Logic and Applications (FPL)}",
	pages		= "140-148",
	publisher	= "IEEE",
	year		= 2024,
	note		= "doi: 10.1109/FPL64840.2024.00028"
}
```
#### If ReducedLUT contributes to your research please also cite:
```bibtex
@inproceedings{reducedlut,
	author = {Cassidy, Oliver and Andronic, Marta and Coward, Samuel and Constantinides, George A.},
	title = "{ReducedLUT: Table Decomposition with ``Don't Care'' Conditions}",
	year = {2025},
	isbn = {9798400713965},
	publisher = {Association for Computing Machinery},
	address = {New York, NY, USA},
	note = "doi: 10.1145/3706628.3708823",
	booktitle = {Proceedings of the 2025 ACM/SIGDA International Symposium on Field Programmable Gate Arrays},
	pages = {36–42},
	location = {Monterey, CA, USA},
}
```
---

## 🤝 Acknowledgements

NeuraLUT builds on foundational work from [LogicNets](https://github.com/Xilinx/logicnets) (Apache 2.0).  
Special thanks to the open-source hardware ML community for their inspiration and contributions.

---