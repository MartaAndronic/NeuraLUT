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

from torchvision import datasets, transforms
from models import MnistNeqModel

configs = {
    "hdr-5l": {
        "hidden_layers": [256, 100, 100, 100],
        "input_bitwidth": 2,
        "hidden_bitwidth": 2,
        "output_bitwidth": 2,
        "input_fanin": 6,
        "hidden_fanin": 6,
        "output_fanin": 6,
        "width_n": 16,
        "weight_decay": 0,
        "batch_size": 128,
        "epochs": 500,
        "learning_rate": 0.003,
        "seed": 8971561,
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

other_options = {
    "cuda": None,
    "log_dir": None,
    "checkpoint": None,
    "device": 1,
}


def train(model, train_cfg, options):
    # Create data loaders for training and inference:
    train_loader = DataLoader(
        datasets.MNIST(
            "mnist_data",
            download=False,
            train=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=train_cfg["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        datasets.MNIST(
            "mnist_data",
            download=False,
            train=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),  # first, convert image to PyTorch tensor
                    transforms.Normalize((0.1307,), (0.3081,)),  # normalize inputs
                ]
            ),
        ),
        batch_size=train_cfg["batch_size"],
        shuffle=False,
    )
    test_loader = DataLoader(
        datasets.MNIST(
            "mnist_data",
            download=False,
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),  # first, convert image to PyTorch tensor
                    transforms.Normalize((0.1307,), (0.3081,)),  # normalize inputs
                ]
            ),
        ),
        batch_size=train_cfg["batch_size"],
        shuffle=False,
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
            data = data.reshape(-1, 784)
            target = torch.nn.functional.one_hot(target, num_classes=10)
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
        data = data.reshape(-1, 784)
        target = torch.nn.functional.one_hot(target, num_classes=10)
        output = model(data)
        pred = output.detach().max(1, keepdim=True)[1]
        if cuda:
            pred = pred.cuda()
        target_label = torch.max(target.detach(), 1, keepdim=True)[1]
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
        default="hdr-5l",
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
        default=False,
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
        metavar="",
        help="A location to store the log output of the training run and the output model (default: %(default)s)",
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

    # Instantiate model

    model_cfg["input_length"] = 784
    model_cfg["output_length"] = 10
    model = MnistNeqModel(model_cfg)
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
            "dataset": "mnist",
        },
    )

    wandb.define_metric("Train Acc (%)", summary="max")
    wandb.define_metric("Test Acc (%)", summary="max")
    wandb.define_metric("Valid Acc(%)", summary="max")
    wandb.define_metric("Train Loss(%)", summary="min")
    wandb.watch(model, log_freq=10)
    train(model, train_cfg, options_cfg)
    wandb.finish()