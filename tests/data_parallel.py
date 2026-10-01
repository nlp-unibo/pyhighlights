"""Fit a model on two CPU processes and compare what each process ends with.

Lightning's ``ddp`` launcher starts the second process by running the script
again, so the fit runs as this script rather than inside pytest. Each process
saves its parameters, and :func:`processes_agree` compares them.
"""

import os
import subprocess
import sys
from pathlib import Path

import torch as th


def processes_agree(key_name: str, directory: Path) -> bool:
    """Whether both processes of a two-process fit hold the same parameters.

    ``key_name`` names a model key in :mod:`pyhighlights.configurations.keys`.
    The fit raises when the model does not run under ``ddp``.
    """
    # One thread per process: pytest already runs one worker per core, and
    # two processes each claiming every core make this test slow.
    environment = {**os.environ, "OMP_NUM_THREADS": "1"}
    subprocess.run(
        [sys.executable, __file__, key_name, str(directory)],
        check=True,
        timeout=300,
        env=environment,
    )
    first = th.load(directory / "rank0.pt")
    second = th.load(directory / "rank1.pt")
    return all(th.equal(a, b) for a, b in zip(first, second, strict=True))


if __name__ == "__main__":
    import lightning as L
    from cinnamon.registry import Registry

    import pyhighlights
    from pyhighlights.components.tasks import SPPTask
    from pyhighlights.configurations import keys

    class SaveParameters(L.Callback):
        def on_train_end(self, trainer, model):
            parameters = [parameter.detach() for parameter in model.parameters()]
            th.save(parameters, directory / f"rank{trainer.global_rank}.pt")

    key = getattr(keys, sys.argv[1])
    directory = Path(sys.argv[2])
    Registry.build(directory=Path(pyhighlights.__file__).parent)
    task = SPPTask(
        loader=keys.TOY,
        model=key,
        save_path=str(directory),
        batch_size=8,
        trainer_args={"accelerator": "cpu", "max_epochs": 1},
    )
    L.Trainer(
        accelerator="cpu",
        devices=2,
        strategy="ddp",
        max_epochs=2,
        limit_train_batches=2,
        limit_val_batches=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[SaveParameters()],
    ).fit(Registry.from_key(key), task.loaders(task.splits())["train"])
