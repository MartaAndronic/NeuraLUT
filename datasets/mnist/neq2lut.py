#  This file is part of NeuraLUT.
#  
#  NeuraLUT is a derivative work based on LogicNets,
#  which is licensed under the Apache License 2.0.

#  Copyright (C) 2021 Xilinx, Inc
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import os
from argparse import ArgumentParser

import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from neuralut.nn import (
    generate_truth_tables,
    lut_inference,
    module_list_to_verilog_module,
)

from train import configs, model_config, test
from models import MnistNeqModel, MnistLutModel
from neuralut.synthesis import synthesize_and_get_resource_counts

other_options = {
    "seed": 3,
    "cuda": None,
    "device": 1,
    "log_dir": None,
    "checkpoint": None,
    "add_registers": False
}

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Synthesize convert a PyTorch trained model into verilog"
    )
    parser.add_argument(
        "--arch",
        type=str,
        choices=configs.keys(),
        default="hdr-5l",
        help="Specific the neural network model to use (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        metavar="N",
        help="Batch size for evaluation (default: %(default)s)",
    )
    parser.add_argument(
        "--input-bitwidth",
        type=int,
        default=None,
        help="Bitwidth to use at the input (default: %(default)s)",
    )
    parser.add_argument(
        "--hidden-bitwidth",
        type=int,
        default=None,
        help="Bitwidth to use for activations in hidden layers (default: %(default)s)",
    )
    parser.add_argument(
        "--output-bitwidth",
        type=int,
        default=None,
        help="Bitwidth to use at the output (default: %(default)s)",
    )
    parser.add_argument(
        "--input-fanin",
        type=int,
        default=None,
        help="Fanin to use at the input (default: %(default)s)",
    )
    parser.add_argument(
        "--hidden-fanin",
        type=int,
        default=None,
        help="Fanin to use for the hidden layers (default: %(default)s)",
    )
    parser.add_argument(
        "--output-fanin",
        type=int,
        default=None,
        help="Fanin to use at the output (default: %(default)s)",
    )
    parser.add_argument(
        "--hidden-layers",
        nargs="+",
        type=int,
        default=None,
        help="A list of hidden layer neuron sizes (default: %(default)s)",
    )
    parser.add_argument(
        "--width_n",
        type=int,
        default=None,
        metavar="",
        help="Width of sub-network(N) (default: %(default)s)",
    )
    parser.add_argument(
        "--clock-period",
        type=float,
        default=1.0,
        help="Target clock frequency to use during Vivado synthesis (default: %(default)s)",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Dataset to use for evaluation (default: %(default)s)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="0",
        help="A location to store the log output of the training run and the output model (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Seed to use for RNG (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        type=int,
        required=True,
        help="Device_id for GPU (default: %(default)s)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="The checkpoint file which contains the model weights",
    )
    parser.add_argument(
        "--add-registers",
        action="store_true",
        default=False,
        help="Add registers between each layer in generated verilog (default: %(default)s)",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        default=True,
        help="Train on a GPU (default: %(default)s)",
    )
    args = parser.parse_args()
    defaults = configs[args.arch]
    options = vars(args)
    del options["arch"]
    config = {}
    for k in options.keys():
        config[k] = (
            options[k] if options[k] is not None else defaults[k]
        )  # Override defaults, if specified.

    if not os.path.exists(config["log_dir"]):
        os.makedirs(config["log_dir"])

    # Split up configuration options to be more understandable
    model_cfg = {}
    for k in model_config.keys():
        model_cfg[k] = config[k]
    options_cfg = {}
    for k in other_options.keys():
        options_cfg[k] = config[k]
    model_cfg["cuda"] = options_cfg["cuda"]
    # Set random seeds
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    os.environ["PYTHONHASHSEED"] = str(config["seed"])
    if options_cfg["cuda"]:
        torch.cuda.manual_seed_all(config["seed"])
        torch.backends.cudnn.deterministic = True
        torch.cuda.set_device(options_cfg["device"])

    # Fetch the test set
    dataset = {}
    dataset[args.dataset_split] = datasets.MNIST(
        "mnist_data",
        download=False,
        train=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),  # first, convert image to PyTorch tensor
                transforms.Normalize((0.1307,), (0.3081,)),  # normalize inputs
            ]
        ),
    )
    test_loader = DataLoader(
        dataset[args.dataset_split], batch_size=config["batch_size"], shuffle=False
    )

    # Instantiate the PyTorch model
    model_cfg["input_length"] = 784
    model_cfg["output_length"] = 10
    model = MnistNeqModel(model_cfg)
    if options_cfg["cuda"]:
        model.cuda()

    # Load the model weights

    checkpoint = torch.load(options_cfg["checkpoint"], map_location="cuda:{}".format(options_cfg["device"]) if options_cfg["cuda"] else "cpu")
    model.load_state_dict(checkpoint["model_dict"])

    # Test the PyTorch model
    print("Running inference on baseline model...")
    baseline_accuracy = test(model, test_loader, cuda=options_cfg["cuda"])
    print("Baseline accuracy: %f" % (baseline_accuracy))

    lut_model = MnistLutModel(model_cfg)
    if options_cfg["cuda"]:
        lut_model.cuda()
    lut_model.load_state_dict(checkpoint['model_dict'])

    # Generate the truth tables in the LUT module
    print("Converting to NEQs to LUTs...")
    generate_truth_tables(lut_model, verbose=True)

    # Test the LUT-based model
    print("Running inference on LUT-based model...")
    lut_inference(lut_model)
    lut_accuracy = test(lut_model, test_loader, cuda=options_cfg["cuda"])
    print("LUT-Based Model accuracy: %f" % (lut_accuracy))
    modelSave = {"model_dict": lut_model.state_dict(), "test_accuracy": lut_accuracy}

    torch.save(modelSave, options_cfg["log_dir"] + "/lut_based_model.pth")
    print("Generating verilog in %s..." % (options_cfg["log_dir"]))
    module_list_to_verilog_module(
        lut_model.module_list,
        "neuralut",
        options_cfg["log_dir"],
        add_registers=options_cfg["add_registers"],
    )
    print("Top level entity stored at: %s/neuralut.v ..." % (options_cfg["log_dir"]))

    io_filename = None

    print("Running inference simulation of Verilog-based model...")
    lut_model.verilog_inference(options_cfg["log_dir"], "neuralut.v", logfile=io_filename, add_registers=options_cfg["add_registers"])
    print("Testing Verilog-Based Model")
    verilog_accuracy = test(lut_model, test_loader, cuda=options_cfg["cuda"])
    print("Verilog-Based Model accuracy: %f" % (verilog_accuracy))