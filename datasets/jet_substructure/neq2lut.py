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

from reducedlut.nn import (
    generate_truth_tables,
    lut_inference,
    logging_inference,
    flush_logs,
    module_list_to_verilog_module,
)

from train import configs, model_config, dataset_config, test
from dataset import JetSubstructureDataset
from models import JetSubstructureNeqModel, JetSubstructureLutModel
from reducedlut.synthesis import synthesize_and_get_resource_counts

from reducedlut.reducedlut import (
    convert_verilog_to_hex,
    remove_verilog_luts,
    run_reducedlut,
    tidy,
)

other_options = {
    "seed": 3,
    "batch_size": 1024,
    "cuda": None,
    "device": 1,
    "log_dir": None,
    "checkpoint": None,
    "add_registers": False,
    "reducedlut": None,
    "exiguity": None,
    "log_bits": None,
}

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Synthesize convert a PyTorch trained model into verilog"
    )
    parser.add_argument(
        "--arch",
        type=str,
        choices=configs.keys(),
        default="jsc-s",
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
        "--dataset-file",
        type=str,
        default="data/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z",
        help="The file to use as the dataset input (default: %(default)s)",
    )
    parser.add_argument(
        "--clock-period",
        type=float,
        default=1.0,
        help="Target clock frequency to use during Vivado synthesis (default: %(default)s)",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="config/yaml_IP_OP_config.yml",
        help="The file to use to configure the input dataset (default: %(default)s)",
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
        default=False,
        help="Train on a GPU (default: %(default)s)",
    )
    parser.add_argument(
        "--reducedlut",
        action="store_true",
        default=False,
        help="Use ReducedLUT (default: %(default)s)",
    )
    parser.add_argument(
        "--exiguity",
        type=int,
        default=250,
        help="Exiguity (default: %(default)s)",
    )
    parser.add_argument(
        "--log_bits",
        action="store_true",
        default=False,
        help="Log ReducedLUT bit compression (default: %(default)s)",
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
    dataset_cfg = {}
    for k in dataset_config.keys():
        dataset_cfg[k] = config[k]
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
    dataset["train"] = JetSubstructureDataset(
        dataset_cfg["dataset_file"], dataset_cfg["dataset_config"], split="train"
    )
    dataset["valid"] = JetSubstructureDataset(
        dataset_cfg["dataset_file"], dataset_cfg["dataset_config"], split="train"
    )  
    dataset["test"] = JetSubstructureDataset(
        dataset_cfg["dataset_file"], dataset_cfg["dataset_config"], split="test"
    )
    train_loader = DataLoader(
        dataset["train"], batch_size=options_cfg["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        dataset["valid"], batch_size=options_cfg["batch_size"], shuffle=False
    )
    test_loader = DataLoader(
        dataset["test"], batch_size=options_cfg["batch_size"], shuffle=False
    )

    # Instantiate the PyTorch model
    x, y = dataset[args.dataset_split][0]
    model_cfg["input_length"] = len(x)
    model_cfg["output_length"] = len(y)
    model = JetSubstructureNeqModel(model_cfg)
    if options_cfg["cuda"]:
        model.cuda()

    # Load the model weights
    checkpoint = torch.load(options_cfg["checkpoint"], map_location="cuda:{}".format(options_cfg["device"]) if options_cfg["cuda"] else "cpu")
    model.load_state_dict(checkpoint["model_dict"])

    # Test the PyTorch model
    print("Running inference on baseline model...")
    baseline_accuracy = test(model, test_loader, cuda=options_cfg["cuda"])
    print("Baseline accuracy: %f" % (baseline_accuracy))

    lut_model = JetSubstructureLutModel(model_cfg)
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

    if options_cfg["reducedlut"]:
        print("Running logging")
        logging_inference(lut_model)
        train_accuracy = test(lut_model, train_loader, cuda=options_cfg["cuda"])
        # train_accuracy = test(lut_model, test_loader, cuda=options_cfg["cuda"])
        print("Flushing logs")
        flush_logs(lut_model, options_cfg["log_dir"])
        logging_inference(lut_model)
        print("Logging complete")

    torch.save(modelSave, options_cfg["log_dir"] + "/lut_based_model.pth")
    print("Generating verilog in %s..." % (options_cfg["log_dir"]))
    module_list_to_verilog_module(
        lut_model.module_list,
        "neuralut",
        options_cfg["log_dir"],
        add_registers=options_cfg["add_registers"],
    )
    print("Top level entity stored at: %s/neuralut.v ..." % (options_cfg["log_dir"]))

    if options_cfg["reducedlut"]:
        print("Running ReducedLUT")
        convert_verilog_to_hex(options_cfg["log_dir"])
        remove_verilog_luts(options_cfg["log_dir"])
        run_reducedlut(options_cfg["log_dir"], options_cfg["exiguity"], options_cfg["log_bits"], 1)
        tidy(options_cfg["log_dir"])

    io_filename = None

    print("Running inference simulation of Verilog-based model...")
    lut_model.verilog_inference(options_cfg["log_dir"], "neuralut.v", logfile=io_filename, add_registers=options_cfg["add_registers"])
    print("Testing Verilog-Based Model")
    verilog_accuracy = test(lut_model, test_loader, cuda=options_cfg["cuda"])
    print("Verilog-Based Model accuracy: %f" % (verilog_accuracy))

    print("Running out-of-context synthesis")
    ret = synthesize_and_get_resource_counts(options_cfg["log_dir"], "neuralut", fpga_part='xcvu9p-flgb2104-2-i', clk_period_ns='1.1', post_synthesis=1)
    print("Max f: " + str(ret))