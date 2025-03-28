# ReducedLUT on the Jet Substructure Classification dataset

To produce models capable of replicating the results found in our paper please follow the steps provided below.

## Download dataset
Navigate to the `datasets/jet_substructure` directory.
```
mkdir -p data
wget https://cernbox.cern.ch/index.php/s/jvFd5MoWhGs1l5v/download -O data/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z
```

## JSC 2L Model
```
python train.py --arch jsc-2l --log_dir jsc-2l --cuda
python neq2lut.py --arch jsc-2l --checkpoint ./test_jsc-2l/best_accuracy.pth --log-dir ./test_jsc-2l/verilog/ --add-registers --seed 8439527 --device 1 --cuda --reducedlut --exiguity 250
```

## JSC 5L Model
```
python train.py --arch jsc-5l --log_dir jsc-5l --cuda
python neq2lut.py --arch jsc-5l --checkpoint ./test_jsc-5l/best_accuracy.pth --log-dir ./test_jsc-5l/verilog/ --add-registers --seed 78439526 --device 1 --cuda --reducedlut --exiguity 250
```


## 📖 Citation
Should you find this work valuable, we kindly request that you consider referencing our papers as below:
```
@inproceedings{10.1145/3706628.3708823,
author = {Cassidy, Oliver and Andronic, Marta and Coward, Samuel and Constantinides, George A.},
title = {ReducedLUT: Table Decomposition with "Don't Care" Conditions},
year = {2025},
isbn = {9798400713965},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3706628.3708823},
doi = {10.1145/3706628.3708823},
booktitle = {Proceedings of the 2025 ACM/SIGDA International Symposium on Field Programmable Gate Arrays},
pages = {36–42},
numpages = {7},
keywords = {compression, hardware acceleration, lookup table, neural network},
location = {Monterey, CA, USA},
series = {FPGA '25}
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