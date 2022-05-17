# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import egg.core as core
import torch
import os


def get_data_opts(parser):
    group = parser.add_argument_group("data")
    
    group.add_argument(
        "--dataset_dir",
        type=str,
        default="/home/usuaris/locatelli/emecom/data",
        help="Dataset location",
    )

    group.add_argument(
        "--image_size", 
        type=int, 
        default=224, 
        help="Image size"
    )

    group.add_argument(
        "--num_workers", 
        type=int, 
        default=4, 
        help="Workers used in the dataloader"
    )
    
def get_gs_opts(parser):
    group = parser.add_argument_group("gumbel softmax")
    
    group.add_argument(
        "--gs_temperature",
        type=float,
        default=5.0,
        help="GS temperature",
    )
    
    group.add_argument(
        "--straight_through",
        default=False,
        action="store_true",
        help="Use straight-through-gs estimator",
    )


def get_vision_module_opts(parser):
    group = parser.add_argument_group("vision module")
    
    group.add_argument(
        "--vision_module_name",
        type=str,
        default="resnet152",
        choices=["resnet50", "resnet101", "resnet152"],
        help="Model name for the vision encoder",
    )
    
    group.add_argument(
        "--pretrained_vision",
        default=True,
        action="store_true",
        help="Use pretrained vision module",
    )

def get_game_arch_opts(parser):
    group = parser.add_argument_group("game architecture")
    
    group.add_argument(
        "--recv_temperature",
        type=float,
        default=1.0,
        help="Temperature for cosine similarity",
    )
    
    group.add_argument(
        "--recv_hidden_dim",
        type=int,
        default=512,
        help="Hidden dim of the distractors projection",
    )
    
    group.add_argument(
        "--embed_dim",
        type=int,
        default=512,
        help="Embedding dim of message and distractors projection",
    )
    
    group.add_argument(
        "--com_len",
        type=int,
        default=1,
        help="Length of messages"
    )


def get_common_opts(params):
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=10e-6,
        help="Weight decay used for SGD",
    )
    
    parser.add_argument(
        "--use_larc", action="store_true", default=False, help="Use LARC optimizer"
    )
    
    parser.add_argument(
        "--pdb",
        action="store_true",
        default=False,
        help="Run the game with pdb enabled",
    )

    get_data_opts(parser)
    get_gs_opts(parser)
    get_vision_module_opts(parser)
    get_game_arch_opts(parser)

    opts = core.init(arg_parser=parser, params=params)
    return opts


def load(game, checkpoint):
    game.load_state_dict(checkpoint.model_state_dict)

def load_from_checkpoint(game, path):
    print(f"# loading trainer state from {path}")
    checkpoint = torch.load(path)
    load(game, checkpoint)


def add_weight_decay(model, weight_decay=1e-5, skip_name=""):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or skip_name in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]
