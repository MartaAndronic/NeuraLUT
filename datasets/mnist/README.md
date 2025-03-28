# ReducedLUT on the MNIST dataset

To produce models capable of replicating the results found in our paper please follow the steps provided below.

## MNIST Model
```
python train.py --cuda
python neq2lut.py --checkpoint ./test_0/best_accuracy.pth --log-dir ./test_0/verilog/ --add-registers --seed 8971561 --device 1 --cuda --reducedlut --exiguity 250
```

## 📖 Citation
Should you find this work valuable, we kindly request that you consider referencing our papers as below:
```
@inproceedings{reducedlut,
author = {Cassidy, Oliver and Andronic, Marta and Coward, Samuel and Constantinides, George A.},
title = "{ReducedLUT: Table Decomposition with ``Don't Care'' Conditions}",
year = {2025},
isbn = {9798400713965},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
note = {doi: 10.1145/3706628.3708823},
booktitle = {Proceedings of the 2025 ACM/SIGDA International Symposium on Field Programmable Gate Arrays},
pages = {36–42},
location = {Monterey, CA, USA},
}
```
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