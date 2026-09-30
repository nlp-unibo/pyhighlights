"""Fit DAR on two CPU processes and save each process's aligner checksum.

``tests/test_dar.py`` runs this as a script. Lightning's ``ddp`` launcher
starts the second process by running the same script again.
"""

import sys
from pathlib import Path

import lightning as L
import torch as th
from cinnamon.registry import Registry

import pyhighlights
from pyhighlights.components.tasks import SPPTask
from pyhighlights.configurations.keys import GRU_DAR, TOY


class AlignerChecksum(L.Callback):
    """Saves what each process's aligner holds once the rationalizer starts."""

    def __init__(self, directory: Path):
        self.directory = directory

    def on_train_epoch_start(self, trainer, model):
        total = sum(parameter.sum() for parameter in model.aligner_parameters())
        th.save(total.detach(), self.directory / f"rank{trainer.global_rank}.pt")


if __name__ == "__main__":
    directory = Path(sys.argv[1])
    Registry.build(directory=Path(pyhighlights.__file__).parent)
    task = SPPTask(
        loader=TOY,
        model=GRU_DAR,
        save_path=str(directory),
        batch_size=8,
        trainer_args={"accelerator": "cpu", "max_epochs": 1},
    )
    loader = task.loaders(task.splits())["train"]
    L.Trainer(
        accelerator="cpu",
        devices=2,
        strategy="ddp",
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[AlignerChecksum(directory)],
    ).fit(Registry.from_key(GRU_DAR, pretrain_epochs=1), loader)
