# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

import json
from pathlib import Path

import egg.core as core
from egg.core import ConsoleLogger

from data import get_dataloader
from game_callbacks import (
    BestStatsTracker,
    DistributedSamplerEpochSetter
)
from games import build_game
from utils import add_weight_decay, get_common_opts


def main(params):
    opts = get_common_opts(params=params)
    if not opts.distributed_context.is_distributed and opts.pdb:
        breakpoint()

    train_loader, val_loader, test_loader = get_dataloader(
        dataset_dir=opts.dataset_dir,
        image_size=opts.image_size,
        batch_size=opts.batch_size,
        num_workers=opts.num_workers,
        seed=opts.random_seed,
    )

    game = build_game(opts)

    model_parameters = add_weight_decay(game, opts.weight_decay, skip_name="bn")

    optimizer = torch.optim.Adam(model_parameters, lr=opts.lr)
    optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=opts.n_epochs
    )

    callbacks = [
        ConsoleLogger(as_json=True, print_train_loss=True),
        BestStatsTracker()
        ]

    if opts.distributed_context.is_distributed:
        callbacks.append(DistributedSamplerEpochSetter())

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        optimizer_scheduler=optimizer_scheduler,
        train_data=train_loader,
        validation_data=val_loader,
        callbacks=callbacks,
        device="cuda",
    )
    trainer.train(n_epochs=opts.n_epochs)

    _, test_interaction = trainer.eval(test_loader)
    if opts.checkpoint_dir:
        output_path = Path(opts.checkpoint_dir)
        output_path.mkdir(exist_ok = True, parents = True)
        torch.save(
            test_interaction, 
            output_path / f"N{opts.com_len}V{opts.vocab_size}_interaction"
        )
        torch.save(
            game, 
            output_path / f"N{opts.com_len}V{opts.vocab_size}_model"
        )
    test_accuracy = test_interaction.aux['acc']
    test_accuracy = float(sum(test_accuracy) / len(test_accuracy))
    print(f"Test accuracy: {test_accuracy:.2f}")
    
    print("GAME OVER")


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    import sys

    main(sys.argv[1:])
