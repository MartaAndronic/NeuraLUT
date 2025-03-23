## NeuraLUT on the MNIST dataset

To reproduce the results in our paper follow the steps below. Subsequently, compile the Verilog files using the following settings (utilize Vivado 2020.1, target the xcvu9p-flgb2104-2-i FPGA part, use the Vivado Flow_PerfOptimized_high settings, and perform synthesis in the Out-of-Context (OOC) mode).

```
python train.py --cuda
python neq2lut.py --checkpoint ./test_0/best_accuracy.pth --log-dir ./test_0/verilog/ --add-registers --seed 8971561 --device 1 --cuda
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