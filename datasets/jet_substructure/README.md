## NeuraLUT on the jet substructure tagging dataset

To reproduce the results in our paper follow the steps below. Subsequently, compile the Verilog files using the following settings (utilize Vivado 2020.1, target the xcvu9p-flgb2104-2-i FPGA part, use the Vivado Flow_PerfOptimized_high settings, and perform synthesis in the Out-of-Context (OOC) mode).

## Download dataset
Navigate to the jet_substructure directory.
```
mkdir -p data
wget https://cernbox.cern.ch/index.php/s/jvFd5MoWhGs1l5v/download -O data/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z
```

```
python train.py --arch jsc-2l --log_dir jsc-2l --cuda
python neq2lut.py --arch jsc-2l --checkpoint ./test_jsc-2l/best_accuracy.pth --log-dir ./test_jsc-2l/verilog/ --add-registers --seed 8766 --device 1 --cuda
```
```
python train.py --arch jsc-5l --log_dir jsc-5l --cuda
python neq2lut.py --arch jsc-5l --checkpoint ./test_jsc-5l/best_accuracy.pth --log-dir ./test_jsc-5l/verilog/ --add-registers --seed 312846 --device 1 --cuda
```


## Citation
Should you find this work valuable, we kindly request that you consider referencing our paper as below:
```
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