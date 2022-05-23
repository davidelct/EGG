# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torchvision
import torch.nn.functional as F
from utils import load_from_checkpoint
from egg.core.gs_wrappers import GumbelSoftmaxWrapper, SymbolReceiverWrapper
from egg.core.interaction import LoggingStrategy
from archs import Sender, Receiver, Game, GameWrapper

def init_vision_module(name, pretrained):    
    modules = {
        "resnet50": torchvision.models.resnet50(pretrained=pretrained),
        "resnet101": torchvision.models.resnet101(pretrained=pretrained),
        "resnet152": torchvision.models.resnet152(pretrained=pretrained),
    }
    if name not in modules:
        raise KeyError(f"{name} is not currently supported.")

    model = modules[name]
    n_features = model.fc.in_features
    model.fc = torch.nn.Identity()

    if pretrained:
        for param in model.parameters():
            param.requires_grad = False
        model = (model.eval())

    return model, n_features

def loss(_x_send, _message, _x_recv, recv_output, _labels, _aux_input):
    labels = torch.arange(recv_output.shape[0], device=recv_output.device)
    acc = (recv_output.argmax(dim=1) == labels).detach().float()
    loss = F.cross_entropy(recv_output, labels, reduction="none")
    return loss, {"acc": acc}

def print_game_config(opts):
    print(f"Game: N{opts.com_len}V{opts.vocab_size}")
    print(f"Vision module: {opts.vision_module_name}")
    print(f"Pretrained: {opts.pretrained_vision}")
    print(f"Image size: {opts.image_size}")
    print(f"Batch size: {opts.batch_size}")
    print(f"Learning rate: {opts.lr}")
    print(f"N epochs: {opts.n_epochs}")
    print(f"GS temperature: {opts.gs_temperature}")
    print(f"Embed dim: {opts.embed_dim}")
    print(f"Recv hidden dim: {opts.recv_hidden_dim}")
    print(f"Recv temperature: {opts.recv_temperature}")

def build_sender_receiver(opts):
    vision_module, in_features = init_vision_module(
        name=opts.vision_module_name, 
        pretrained=opts.pretrained_vision
    )

    sender = Sender(
                vision_module=vision_module,
                input_dim=in_features,
                vocab_size=opts.vocab_size,
                embed_dim=opts.embed_dim,
                com_len=opts.com_len,
                temperature=opts.gs_temperature,
                straight_through=opts.straight_through,
    )
    
    receiver = Receiver(
                vision_module=vision_module,
                input_dim=in_features,
                hidden_dim=opts.recv_hidden_dim,
                embed_dim=opts.embed_dim,
                com_len=opts.com_len,
                temperature=opts.recv_temperature
    )
    
    return sender, receiver

def build_game(opts):
    print_game_config(opts)
    
    train_logging_strategy = LoggingStrategy(
        store_sender_input = False,
        store_receiver_input = False,
        store_labels = False,
        store_aux_input = False,
        store_message = False,
        store_receiver_output = False,
        store_message_length = False
    )
    
    test_logging_strategy = LoggingStrategy(
        store_sender_input = True,
        store_receiver_input = True,
        store_labels = True,
        store_aux_input = True,
        store_message = True,
        store_receiver_output = True,
        store_message_length = False
    )
    
    sender, receiver = build_sender_receiver(opts)

    game = GameWrapper(
        game=Game(train_logging_strategy,test_logging_strategy),
        sender=sender,
        receiver=receiver,
        loss=loss
    )

    if opts.distributed_context.is_distributed:
        game = torch.nn.SyncBatchNorm.convert_sync_batchnorm(game)

    return game
