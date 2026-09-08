"""Tasks: one experiment, start to finish.

A task is what a paper's table row is made of. It names the corpus, the
preprocessing, the model and the metrics as registration keys, runs the whole
thing over a list of seeds, and writes down what happened -- so reproducing a
number means running one key, not remembering which loader went with which
checkpoint.

Seeds are a list rather than a number because a single run of a
select-then-predict model says very little: the selector is trained through a
discrete choice, and the spread across seeds is part of the result.
"""

from __future__ import annotations

import abc
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import lightning as L
import numpy as np
import pandas as pd
import torch as th
from cinnamon.registry import RegistrationKey, Registry
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from pyhighlights.components.data import (
    HighlightCollator,
    HighlightDataset,
    HighlightTokenizer,
    HuggingFaceTokenizer,
    VocabularyTokenizer,
)
from pyhighlights.components.loaders import HighlightLoader, to_examples
from pyhighlights.components.models.base import Model
from pyhighlights.components.preprocessors import Preprocessor
from pyhighlights.utility.metrics import BoundMetric

logger = logging.getLogger(__name__)

__all__ = ["SPPTask", "Task", "summarize", "vocabulary"]


def vocabulary(frames: Iterable[pd.DataFrame], size: int) -> Dict[str, int]:
    """Token ids ``1`` to ``size - 1``, most frequent first.

    ``size`` counts the ids in use rather than the tokens named, because it is
    the embedding's width that has to hold them: id ``0`` is the unknown and
    padding token, so ``size - 1`` tokens are mapped and the rest fall back to
    it.

    Built from the training split alone. A vocabulary fitted on evaluation text
    would leak it -- quietly, since nothing downstream can tell where an id
    came from.
    """
    if size < 2:
        raise ValueError("a vocabulary needs room for the unknown token and one more")

    counts: Counter = Counter()
    for frame in frames:
        for tokens in frame["tokens"]:
            counts.update(tokens)
    return {
        token: index
        for index, (token, _) in enumerate(counts.most_common(size - 1), start=1)
    }


def summarize(runs: Sequence[Mapping[str, float]]) -> Dict[str, Dict[str, float]]:
    """Mean and standard deviation of each metric across seeds."""
    names = sorted({name for run in runs for name in run})
    return {
        name: {
            "mean": float(np.mean([run[name] for run in runs if name in run])),
            "std": float(np.std([run[name] for run in runs if name in run])),
            "values": [float(run[name]) for run in runs if name in run],
        }
        for name in names
    }


class Task(abc.ABC):
    """One experiment, run over a list of seeds and written down."""

    def __init__(
        self,
        name: str = "task",
        save_path: str | Path | None = None,
    ):
        self.name = name
        self.save_path = Path(save_path) if save_path is not None else Path("results")

    @property
    def directory(self) -> Path:
        return self.save_path / self.name

    @abc.abstractmethod
    def run(self) -> Dict[str, Any]:
        """Run the experiment and return its results."""

    def serialize(self, results: Mapping[str, Any]) -> Path:
        """Write the results and the settings that produced them."""
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "results.json").write_text(json.dumps(results, indent=2))
        (self.directory / "config.json").write_text(
            json.dumps({key: str(value) for key, value in vars(self).items()}, indent=2)
        )
        return self.directory


class SPPTask(Task):
    """Trains a select-then-predict model over a corpus, once per seed.

    The pieces are registration keys, so the same task definition swaps its
    corpus or its model without touching code. ``preprocessor`` is optional
    only for a corpus that needs none; HateXplain has no label until one has
    run, and the loader says so rather than guessing.

    Each seed trains from scratch, restores the checkpoint that scored best on
    validation, and is evaluated on validation and test. What lands on disk is
    ``results.json`` -- every seed's metrics, plus their mean and standard
    deviation -- ``config.json``, and, when asked, the test predictions.
    """

    def __init__(
        self,
        loader: RegistrationKey[HighlightLoader],
        model: RegistrationKey[Model],
        preprocessor: RegistrationKey[Preprocessor] | None = None,
        train_metrics: List[RegistrationKey[BoundMetric]] | None = None,
        val_metrics: List[RegistrationKey[BoundMetric]] | None = None,
        test_metrics: List[RegistrationKey[BoundMetric]] | None = None,
        seeds: Sequence[int] = (42,),
        batch_size: int = 32,
        max_length: int | None = None,
        vocabulary_size: int = 10_000,
        pretrained_model_card: str | None = None,
        monitor: str = "val_loss",
        patience: int = 5,
        store_predictions: bool = False,
        trainer_args: Mapping[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not seeds:
            raise ValueError("a task needs at least one seed")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.loader = loader
        self.model = model
        self.preprocessor = preprocessor
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics
        self.seeds = list(seeds)
        self.batch_size = batch_size
        self.max_length = max_length
        self.vocabulary_size = vocabulary_size
        self.pretrained_model_card = pretrained_model_card
        self.monitor = monitor
        self.patience = patience
        self.store_predictions = store_predictions
        # Defaults a batch run wants, overridden by whatever the caller passes:
        # the progress bar writes one line per step into a log nobody reads.
        self.trainer_args = {
            "accelerator": "cpu",
            "max_epochs": 5,
            "enable_progress_bar": False,
            "enable_model_summary": False,
            **dict(trainer_args or {}),
        }

    def splits(self) -> Dict[str, pd.DataFrame]:
        """The corpus, loaded and preprocessed."""
        splits = Registry.from_key(self.loader, expected_type=HighlightLoader).load()
        if self.preprocessor is not None:
            splits = Registry.from_key(
                self.preprocessor, expected_type=Preprocessor
            ).process(splits)
        return splits

    def tokenizer(self, splits: Mapping[str, pd.DataFrame]) -> HighlightTokenizer:
        """A subword tokenizer when a model card is named, else a vocabulary.

        The vocabulary is fitted on ``train`` only, and its size has to match
        the backbone's ``vocab_size``: an id the embedding has no row for is a
        crash at the first batch.
        """
        if self.pretrained_model_card is not None:
            return HuggingFaceTokenizer(self.pretrained_model_card)
        return VocabularyTokenizer(
            vocabulary(
                [splits[name] for name in ("train",) if name in splits],
                size=self.vocabulary_size,
            )
        )

    def loaders(self, splits: Mapping[str, pd.DataFrame]) -> Dict[str, DataLoader]:
        collator = HighlightCollator(self.tokenizer(splits), self.max_length)
        return {
            name: DataLoader(
                HighlightDataset(to_examples(frame)),
                batch_size=self.batch_size,
                shuffle=name == "train",
                collate_fn=collator,
            )
            for name, frame in splits.items()
        }

    def build_model(self) -> Model:
        return Registry.from_key(
            self.model,
            expected_type=Model,
            train_metrics=self.train_metrics,
            val_metrics=self.val_metrics,
            test_metrics=self.test_metrics,
        )

    def fit(self, seed: int, loaders: Mapping[str, DataLoader]) -> Dict[str, float]:
        """Train one model and score it, leaving its checkpoint behind."""
        seed_everything(seed=seed, workers=True)
        model = self.build_model()

        checkpoints = self.directory / f"seed={seed}"
        checkpoints.mkdir(parents=True, exist_ok=True)
        checkpoint = ModelCheckpoint(
            monitor=self.monitor, mode="min", dirpath=checkpoints
        )
        trainer = L.Trainer(
            **{"default_root_dir": checkpoints, **self.trainer_args},
            callbacks=[
                EarlyStopping(monitor=self.monitor, mode="min", patience=self.patience),
                checkpoint,
            ],
        )
        trainer.fit(
            model,
            train_dataloaders=loaders["train"],
            val_dataloaders=loaders.get("val"),
        )

        # The last epoch is not the best one: early stopping stops after
        # `patience` worse epochs, so scoring the weights still in memory
        # reports a model nobody would have kept.
        if not checkpoint.best_model_path:
            # Nothing was ever checkpointed: `monitor` names a metric no split
            # logs. Scoring the weights training happened to end on is not the
            # same experiment, so say which metric is missing.
            logger.warning(
                "%s: nothing monitored %s, so seed %s is scored on its last "
                "epoch rather than its best one",
                self.name,
                self.monitor,
                seed,
            )
        else:
            # A checkpoint stores the model's hyperparameters, and those hold
            # registration keys, whose tags are a frozenset. Allowlisting those
            # three keeps the load in weights-only mode rather than unpickling
            # whatever a checkpoint file happens to contain.
            with th.serialization.safe_globals([RegistrationKey, frozenset, set]):
                state = th.load(checkpoint.best_model_path, map_location="cpu")
            model.load_state_dict(state["state_dict"])

        results: Dict[str, float] = {}
        if "val" in loaders:
            results.update(trainer.validate(model, dataloaders=loaders["val"])[0])
        if "test" in loaders:
            if self.store_predictions:
                model.enable_storing_predictions()
            results.update(trainer.test(model, dataloaders=loaders["test"])[0])
            if self.store_predictions:
                pd.to_pickle(model.predictions, checkpoints / "predictions.pkl")
                model.flush_predictions()
                model.disable_storing_predictions()
        return results

    def run(self) -> Dict[str, Any]:
        loaders = self.loaders(self.splits())
        runs = [self.fit(seed, loaders) for seed in self.seeds]

        results = {
            "name": self.name,
            "seeds": self.seeds,
            "runs": [
                {name: float(value) for name, value in run.items()} for run in runs
            ],
            "summary": summarize(runs),
        }
        logger.info("%s: %s", self.name, json.dumps(results["summary"], indent=2))
        self.serialize(results)
        return results
