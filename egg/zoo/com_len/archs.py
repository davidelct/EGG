# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from egg.core.gs_wrappers import GumbelSoftmaxLayer, RelaxedEmbedding
from egg.core.interaction import LoggingStrategy

class Sender(nn.Module):
    def __init__(self, vision_module, input_dim, vocab_size, embed_dim, com_len, 
                 temperature, straight_through):
        super(Sender, self).__init__()
        
        self.vision_module = vision_module
        
        self.com_module = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, vocab_size),
                nn.BatchNorm1d(vocab_size),
                GumbelSoftmaxLayer(
                    temperature=temperature, 
                    straight_through=straight_through
                )
            ) for _ in range(com_len)]
        )
        
        self.embed_module = nn.ModuleList([
            RelaxedEmbedding(vocab_size, embed_dim)
            for _ in range(com_len)]
        )
        self.com_len = com_len
        
    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            if module != self.vision_module: module.train(mode)
        return self

    def forward(self, x, aux_input=None):
        vision_module_out = self.vision_module(x)
        
        message = [layer(vision_module_out) for layer in self.com_module]
        embedded_message = [self.embed_module[i](message[i]) for i in range(self.com_len)]
        
        message = torch.cat(message, dim=1)                              
        embedded_message = torch.cat(embedded_message, dim=1)
        
        return embedded_message, message


class Receiver(nn.Module):
    def __init__(self, vision_module, input_dim, hidden_dim, embed_dim, com_len, 
                 temperature):
        super(Receiver, self).__init__()
        
        self.vision_module = vision_module
        
        self.fc = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, embed_dim, bias = False),
            ) for _ in range(com_len)]
        )
        
        self.com_len = com_len
        self.temperature = temperature
        
    
    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            if module != self.vision_module:
                module.train(mode)
        return self

    def forward(self, message, distractors, aux_input=None):
        vision_module_out = self.vision_module(distractors)
        
        distractors = [
            self.fc[i](vision_module_out)
            for i in range(self.com_len)
        ]
        
        distractors = torch.cat(distractors, dim=1)
        
        similarity_scores = (
            torch.nn.functional.cosine_similarity(
                message.unsqueeze(1), distractors.unsqueeze(0), dim=2
            )
            / self.temperature
        )

        return similarity_scores


class Game(nn.Module):
    def __init__(self, train_logging_strategy=None, test_logging_strategy=None):
        super(Game, self).__init__()

        self.train_logging_strategy = (
            LoggingStrategy()
            if train_logging_strategy is None
            else train_logging_strategy
        )
        self.test_logging_strategy = (
            LoggingStrategy()
            if test_logging_strategy is None
            else test_logging_strategy
        )

    def forward(self, sender, receiver, loss, sender_input, labels, 
                receiver_input=None, aux_input=None):
        
        embedded_message, message = sender(sender_input, aux_input)
        receiver_output = receiver(embedded_message, receiver_input, aux_input)

        loss, aux_info = loss(sender_input, embedded_message, receiver_input,
            receiver_output, labels, aux_input
        )
        
        logging_strategy = (self.train_logging_strategy if self.training 
                            else self.test_logging_strategy
        )

        interaction = logging_strategy.filtered_interaction(
            sender_input=sender_input,
            receiver_input=receiver_input.detach(),
            labels=labels.detach(),
            aux_input=aux_input,
            receiver_output=receiver_output.detach(),
            message=message.detach(),
            message_length=None,
            aux=aux_info,
        )
        return loss.mean(), interaction

class GameWrapper(nn.Module):
    def __init__(self, game, sender, receiver, loss):
        super().__init__()

        self.game = game
        self.sender = sender
        self.receiver = receiver
        self.loss = loss
        self.device = "cuda"

    def forward(self, *args, **kwargs):
        mean_loss, interactions = self.game(
            self.sender.to(self.device), 
            self.receiver.to(self.device), 
            self.loss, *args, **kwargs
        )

        return mean_loss, interactions 
