## NeuraLUT-Assemble on the jet substructure tagging dataset (CERNBox)

This folder provides the code and resources to reproduce our NeuraLUT-Assemble results on the CERNBox jet substructure tagging dataset.

We also include a pretrained checkpoint in the test_demo folder so you can skip training and go straight to evaluation and hardware generation.
>These checkpoints are not the exact ones used in the paper but are provided for convenience and practice.

## Download JSC dataset from CERNBox
Navigate to the jet_substructure directory.
```
mkdir -p data
wget https://cernbox.cern.ch/index.php/s/jvFd5MoWhGs1l5v/download -O data/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z
```

### 📓 Demo Notebook
For a quick and interactive overview, check out demo.ipynb.

This notebook:

* Loads the pretrained checkpoint
* Verifies the test accuracy
* Generates the truth tables
* Runs a software simulation on the truth tables to validate accuracy

Generates Verilog files (⚠️ Note: only software simulation is performed in the notebook)

For full hardware simulation and Verilog compilation, please use neq2lut.py as shown below.

### 🚀 Quickstart

To reproduce the full results, including hardware simulation with Verilator, follow these steps:

1. Train the Model (optional)
```
python train.py --arch jsc-cernbox --log_dir demo --cuda --device 1
```
2. Convert to Verilog, Simulate, and Evaluate
This script:
* Loads the trained checkpoint
* Verifies test accuracy
* Generates truth tables
* Runs both software simulation and hardware simulation using Verilator
* Compiles Verilog files for FPGA inference

```
python neq2lut.py --arch jsc-cernbox \
                  --checkpoint ./test_demo/best_accuracy.pth \
                  --log-dir ./test_demo/verilog/ \
                  --add-registers \
                  --device 1 \
                  --imask ./test_demo/imask.pth \
                  --cuda
```


## 📖 Citation
Should you find this work valuable, we kindly request that you consider referencing our papers as below:
```bibtex
@inproceedings{andronic2025neuralut-assemble,
	author	= "Andronic, Marta and Constantinides, George A.",
	title		= "{NeuraLUT-Assemble: Hardware-Aware Assembling of Sub-Neural Networks for Efficient LUT Inference}",
	booktitle	= "{2025 IEEE 33rd Annual International Symposium on Field-Programmable Custom Computing Machines (FCCM)}",
	pages		= "208-216",
	publisher	= "IEEE",
	year		= 2025,
	note		= "doi: 10.1109/FCCM62733.2025.00077"
}
```
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
```bibtex
@inproceedings{andronic2024neuralut,
	author	= "Andronic, Marta and Constantinides, George A.",
	title		= "{PolyLUT: Ultra-Low Latency Polynomial Inference With Hardware-Aware Structured Pruning}",
	booktitle	= "{IEEE Transactions on Computers}",
	pages		= "3181-3194",
	publisher	= "IEEE",
	year		= 2025,
	note		= "doi: 10.1109/TC.2025.3586311"
}
```