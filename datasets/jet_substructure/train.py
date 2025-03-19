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
from functools import reduce
import random

import numpy as np
import wandb
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import JetSubstructureDataset
from models import JetSubstructureNeqModel

configs = {
    "jsc-2l": {
        "hidden_layers": [32],
        "input_bitwidth": 4,
        "hidden_bitwidth": 4,
        "output_bitwidth": 4,
        "input_fanin": 3,
        "hidden_fanin": 3,
        "output_fanin": 3,
        "width_n": 8,
        "weight_decay": 0,
        "batch_size": 1024,
        "epochs": 1000,
        "learning_rate": 1e-3,
        "seed": 8766,
        "checkpoint": None,
    },
    "jsc-5l": {
        "hidden_layers": [128, 128, 128, 64],
        "input_bitwidth": 7,
        "hidden_bitwidth": 4,
        "output_bitwidth": 4,
        "input_fanin": 2,
        "hidden_fanin": 3,
        "output_fanin": 3,
        "width_n": 16,
        "weight_decay": 0,
        "batch_size": 1024,
        "epochs": 1000,
        "learning_rate": 1e-3,
        "seed": 312846,
        "checkpoint": None,
    },
}

# A dictionary, so we can set some defaults if necessary
model_config = {
    "hidden_layers": None,
    "input_bitwidth": None,
    "hidden_bitwidth": None,
    "output_bitwidth": None,
    "input_fanin": None,
    "hidden_fanin": None,
    "output_fanin": None,
    "width_n": None,
}

training_config = {
    "weight_decay": None,
    "batch_size": None,
    "epochs": None,
    "learning_rate": None,
    "seed": None,
}

dataset_config = {
    "dataset_file": None,
    "dataset_config": None,
}

other_options = {
    "cuda": None,
    "log_dir": None,
    "checkpoint": None,
    "device": 1,
}


def train(model, datasets, train_cfg, options):
    # Create data loaders for training and inference:
    train_loader = DataLoader(
        datasets["train"], batch_size=train_cfg["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        datasets["valid"], batch_size=train_cfg["batch_size"], shuffle=False
    )
    test_loader = DataLoader(
        datasets["test"], batch_size=train_cfg["batch_size"], shuffle=False
    )

    # Configure optimizer
    weight_decay = train_cfg["weight_decay"]
    decay_exclusions = [
        "bn",
        "bias",
        "learned_value",
    ]  # Make a list of parameters name fragments which will ignore weight decay TODO: make this list part of the train_cfg
    decay_params = []
    no_decay_params = []
    for pname, params in model.named_parameters():
        if params.requires_grad:
            if reduce(
                lambda a, b: a or b, map(lambda x: x in pname, decay_exclusions)
            ):  # check if the current label should be excluded from weight decay
                # print("Disabling weight decay for %s" % (pname))
                no_decay_params.append(params)
            else:
                # print("Enabling weight decay for %s" % (pname))
                decay_params.append(params)
        # else:
        # print("Ignoring %s" % (pname))
    params = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = optim.AdamW(
        params,
        lr=train_cfg["learning_rate"],
        betas=(0.5, 0.999),
        weight_decay=weight_decay,
    )

    # Configure scheduler
    steps = len(train_loader)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=steps * 100, T_mult=1
    )

    # Configure criterion
    criterion = nn.CrossEntropyLoss()

    # Push the model to the GPU, if necessary
    if options["cuda"]:
        model.cuda()

    # Main training loop
    maxAcc = 0.0
    num_epochs = train_cfg["epochs"]
    for epoch in range(0, num_epochs):
        # Train for this epoch
        model.train()
        accLoss = 0.0
        correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            if options["cuda"]:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, torch.max(target, 1)[1])
            pred = output.detach().max(1, keepdim=True)[1]
            target_label = torch.max(target.detach(), 1, keepdim=True)[1]
            curCorrect = pred.eq(target_label).long().sum()
            curAcc = 100.0 * curCorrect / len(data)
            correct += curCorrect
            accLoss += loss.detach() * len(data)
            loss.backward()
            optimizer.step()
            scheduler.step()

        accLoss /= len(train_loader.dataset)
        accuracy = 100.0 * correct / len(train_loader.dataset)
        val_accuracy = test(model, val_loader, options["cuda"])
        test_accuracy = test(model, test_loader, options["cuda"])
        modelSave = {
            "model_dict": model.state_dict(),
            "optim_dict": optimizer.state_dict(),
            "val_accuracy": val_accuracy,
            "test_accuracy": test_accuracy,
            "epoch": epoch,
        }
        torch.save(modelSave, "test_" + options["log_dir"] + "/checkpoint.pth")
        if maxAcc < test_accuracy:
            torch.save(modelSave, "test_" + options["log_dir"] + "/best_accuracy.pth")
            maxAcc = test_accuracy

        wandb.log(
            {
                "Train Acc (%)": accuracy.detach().cpu().numpy(),
                "Train Loss(%)": accLoss.detach().cpu().numpy(),
                "Test Acc (%)": test_accuracy,
                "Valid Acc(%)": val_accuracy,
            }
        )


def test(model, dataset_loader, cuda):
    model.eval()
    correct = 0
    accLoss = 0.0
    for batch_idx, (data, target) in enumerate(dataset_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        pred = output.detach().max(1, keepdim=True)[1]
        target_label = torch.max(target.detach(), 1, keepdim=True)[1]
        if cuda:
            pred = pred.cuda()
        curCorrect = pred.eq(target_label).long().sum()
        curAcc = 100.0 * curCorrect / len(data)
        correct += curCorrect
    accuracy = 100 * float(correct) / len(dataset_loader.dataset)
    return accuracy


if __name__ == "__main__":
    parser = ArgumentParser(description="NeuraLUT Example")
    parser.add_argument(
        "--arch",
        type=str,
        choices=configs.keys(),
        default="jsc-2l",
        metavar="",
        help="Specific the neural network model to use (default: %(default)s)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=None,
        metavar="",
        help="Weight decay (default: %(default)s)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        metavar="",
        help="Batch size for training (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        metavar="",
        help="Number of epochs to train (default: %(default)s)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        metavar="",
        help="Initial learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        default=True,
        help="Train on a GPU (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="",
        help="Seed to use for RNG (default: %(default)s)",
    )
    parser.add_argument(
        "--input_bitwidth",
        type=int,
        default=None,
        metavar="",
        help="Bitwidth to use at the input (default: %(default)s)",
    )
    parser.add_argument(
        "--hidden_bitwidth",
        type=int,
        default=None,
        metavar="",
        help="Bitwidth to use for activations in hidden layers (default: %(default)s)",
    )
    parser.add_argument(
        "--output_bitwidth",
        type=int,
        default=None,
        metavar="",
        help="Bitwidth to use at the output (default: %(default)s)",
    )
    parser.add_argument(
        "--input_fanin",
        type=int,
        default=None,
        metavar="",
        help="Fanin to use at the input (default: %(default)s)",
    )
    parser.add_argument(
        "--hidden_fanin",
        type=int,
        default=None,
        metavar="",
        help="Fanin to use for the hidden layers (default: %(default)s)",
    )
    parser.add_argument(
        "--output_fanin",
        type=int,
        default=None,
        metavar="",
        help="Fanin to use at the output (default: %(default)s)",
    )
    parser.add_argument(
        "--width_n",
        type=int,
        default=None,
        metavar="",
        help="Width of sub-network(N) (default: %(default)s)",
    )
    parser.add_argument(
        "--hidden_layers",
        nargs="+",
        type=int,
        default=None,
        metavar="",
        help="A list of hidden layer neuron sizes (default: %(default)s)",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="0",
        help="A location to store the log output of the training run and the output model (default: %(default)s)",
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        default="data/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z",
        metavar="",
        help="The file to use as the dataset input (default: %(default)s)",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="config/yaml_IP_OP_config.yml",
        metavar="",
        help="The file to use to configure the input dataset (default: %(default)s)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        metavar="",
        help="Retrain the model from a previous checkpoint (default: %(default)s)",
    )
    parser.add_argument(
        "--device", 
        type=int, 
        default=1, 
        metavar="", 
        help="Device_id for GPU",
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

    if not os.path.exists("test_" + config["log_dir"]):
        os.makedirs("test_" + config["log_dir"])

    # Split up configuration options to be more understandable
    model_cfg = {}
    for k in model_config.keys():
        model_cfg[k] = config[k]
    train_cfg = {}
    for k in training_config.keys():
        train_cfg[k] = config[k]
    dataset_cfg = {}
    for k in dataset_config.keys():
        dataset_cfg[k] = config[k]
    options_cfg = {}
    for k in other_options.keys():
        options_cfg[k] = config[k]
    model_cfg["cuda"] = options_cfg["cuda"]

    # Set random seeds
    random.seed(train_cfg["seed"])
    np.random.seed(train_cfg["seed"])
    torch.manual_seed(train_cfg["seed"])
    os.environ["PYTHONHASHSEED"] = str(train_cfg["seed"])
    if options["cuda"]:
        torch.cuda.manual_seed_all(train_cfg["seed"])
        torch.backends.cudnn.deterministic = True
        torch.cuda.set_device(options_cfg["device"])

    # Fetch the datasets
    dataset = {}
    dataset["train"] = JetSubstructureDataset(
        dataset_cfg["dataset_file"], dataset_cfg["dataset_config"], split="train"
    )
    dataset["valid"] = JetSubstructureDataset(
        dataset_cfg["dataset_file"], dataset_cfg["dataset_config"], split="train"
    )  # This dataset is so small, we'll just use the training set as the validation set, otherwise we may have too few trainings examples to converge.
    dataset["test"] = JetSubstructureDataset(
        dataset_cfg["dataset_file"], dataset_cfg["dataset_config"], split="test"
    )

    # Instantiate model

    x, y = dataset["train"][0]
    model_cfg["input_length"] = len(x)
    model_cfg["output_length"] = len(y)
    model = JetSubstructureNeqModel(model_cfg)
    if options_cfg["checkpoint"] is not None:
        print(f"Loading pre-trained checkpoint {options_cfg['checkpoint']}")
        checkpoint = torch.load(options_cfg["checkpoint"], map_location="cpu")
        model.load_state_dict(checkpoint["model_dict"])

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="NeuraLUT",
        # track hyperparameters and run metadata
        config={
            "hidden_layers": model_cfg["hidden_layers"],
            "input_bitwidth": model_cfg["input_bitwidth"],
            "hidden_bitwidth": model_cfg["hidden_bitwidth"],
            "output_bitwidth": model_cfg["output_bitwidth"],
            "input_fanin": model_cfg["input_fanin"],
            "hidden_fanin": model_cfg["hidden_fanin"],
            "output_fanin": model_cfg["output_fanin"],
            "width_n": model_cfg["width_n"],
            "weight_decay": train_cfg["weight_decay"],
            "batch_size": train_cfg["batch_size"],
            "epochs": train_cfg["epochs"],
            "learning_rate": train_cfg["learning_rate"],
            "seed": train_cfg["seed"],
            "dataset": "jsc",
        },
    )

    wandb.define_metric("Train Acc (%)", summary="max")
    wandb.define_metric("Test Acc (%)", summary="max")
    wandb.define_metric("Valid Acc(%)", summary="max")
    wandb.define_metric("Train Loss(%)", summary="min")
    wandb.watch(model, log_freq=10)
    train(model, dataset, train_cfg, options_cfg)
    wandb.finish()