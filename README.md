# NeuraLUT - COMING SOON
NeuraLUT: Hiding Neural Network Density in Boolean Synthesizable Functions

NeuraLUT is the first quantized neural network training methodology that maps dense and full-precision sub-networks with skip-connections to LUTs to leverage the underlying structure of the FPGA architecture. This project is a derivative work based on LogicNets (https://github.com/Xilinx/logicnets) which is licensed under the Apache License 2.0.

## Setup
**Install Vivado Design Suite**

**Create a Conda environment**

Requirements:
* python=3.8
* pytorch==1.4.0
* torchvision

## Install Brevitas
```
conda install -y packaging pyparsing
conda install -y docrep -c conda-forge
pip install --no-cache-dir git+https://github.com/Xilinx/brevitas.git@67be9b58c1c63d3923cac430ade2552d0db67ba5
```

## Install PolyLUT package
```
cd NeuraLUT
pip install .
```
## Install wandb + login
```
pip install wandb
wandb.login()
```
## Summary of major modifications from LogicNets
* We present a novel way of designing deep NNs with specific sparsity patterns that resemble sparsely connected dense partitions, enabling the encapsulation of sub-networks
entirely within a single LUT. We enhance the training by integrating skip-connections
in our sub-networks which facilitate the flow of gradients,
promoting stable and efficient learning.
* Both NeuraLUT and LogicNets are capable of training on the Jet Substructure Tagging dataset. Additionally, NeuraLUT offers compatibility with the MNIST dataset.
* Introducing novel model architectures, NeuraLUT's distinct structures are detailed in our accompanying paper.
* NeuraLUT is tailored for optimal GPU utilization.
* To track experiments NeuraLUT uses WandB insetad of TensorBoard.
* While LogicNets enforces an a priori sparsity by utilizing a weight mask that deactivates specific weights, NeuraLUT takes a different approach. It doesn't employ a weight mask but rather utilizes a feature mask (FeatureMask), which reshapes the feature vector to incorporate only fanin features per output channel.
* NeuraLUT introduces a completely new forward function that contains multiple fully-connected layers with skip-connections.
* The function "calculate_truth_tables" was adapted to align with the NeuraLUT neuron structure, and it was also improved for efficiency.

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