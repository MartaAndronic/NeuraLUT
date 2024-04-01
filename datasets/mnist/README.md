## NeuraLUT on the MNIST dataset

To reproduce the results in our paper follow the steps below. Subsequently, compile the Verilog files using the following settings (utilize Vivado 2020.1, target the xcvu9p-flgb2104-2-i FPGA part, use the Vivado Flow_PerfOptimized_high settings, and perform synthesis in the Out-of-Context (OOC) mode).

```
python train.py
python neq2lut.py --checkpoint ./test_0/best_accuracy.pth --log-dir ./test_0/verilog/ --add-registers --seed 8971561 --device 1
```


## Citation
Should you find this work valuable, we kindly request that you consider referencing our paper as below:
```
@misc{andronic2024neuralut,
      title={NeuraLUT: Hiding Neural Network Density in Boolean Synthesizable Functions}, 
      author={Marta Andronic and George A. Constantinides},
      year={2024},
      eprint={2403.00849},
      archivePrefix={arXiv},
      primaryClass={cs.AR}
}
```