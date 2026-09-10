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
import itertools
import json
import logging
from collections import Counter
from datetime import datetime
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

from pyhighlights.components import faithfulness
from pyhighlights.components.data import (
    HighlightCollator,
    HighlightDataset,
    HighlightTokenizer,
    HuggingFaceTokenizer,
    VocabularyTokenizer,
)
from pyhighlights.components.loaders import HighlightLoader, to_examples
from pyhighlights.components.models.base import Model
from pyhighlights.components.models.spp.genspp import GenSPPTrainer
from pyhighlights.components.preprocessors import ClassWeights, Preprocessor
from pyhighlights.utility import manifest
from pyhighlights.utility.embeddings import load_vectors
from pyhighlights.utility.losses import Loss
from pyhighlights.utility.metrics import BoundMetric, build_metrics

logger = logging.getLogger(__name__)

__all__ = [
    "ClassWeightsTask",
    "GenSPPTask",
    "SPPTask",
    "Task",
    "load_splits",
    "summarize",
    "vocabulary",
]


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


def load_splits(
    loader: RegistrationKey[HighlightLoader],
    preprocessor: RegistrationKey[Preprocessor] | None = None,
) -> Dict[str, pd.DataFrame]:
    """The corpus a key names, preprocessed by the key that names how."""
    splits = Registry.from_key(loader, expected_type=HighlightLoader).load()
    if preprocessor is not None:
        splits = Registry.from_key(preprocessor, expected_type=Preprocessor).process(
            splits
        )
    return splits


def summarize(runs: Sequence[Mapping[str, float]]) -> Dict[str, Dict[str, float]]:
    """Mean and standard deviation of each metric across seeds."""
    names = sorted({name for run in runs for name in run})
    found = {name: [float(run[name]) for run in runs if name in run] for name in names}
    return {
        name: {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "values": values,
        }
        for name, values in found.items()
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
        self._started: str | None = None

    @property
    def directory(self) -> Path:
        """Where this run writes, stamped with the moment it started.

        One directory per run, never reused: two runs of the same task are two
        results to compare, and the second quietly replacing the first is a
        measurement lost to a re-run somebody forgot they had already done. The
        stamp is taken once and kept, so every seed, the metrics and the
        manifest land together.
        """
        if self._started is None:
            stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            # Two runs inside one second would otherwise share a directory,
            # which is the one thing the stamp is here to prevent.
            self._started = stamp
            for suffix in itertools.count(2):
                if not (self.save_path / self.name / self._started).exists():
                    break
                self._started = f"{stamp}-{suffix}"
        return self.save_path / self.name / self._started

    @abc.abstractmethod
    def run(self) -> Dict[str, Any]:
        """Run the experiment and return its results."""

    def serialize(self, results: Mapping[str, Any]) -> Path:
        """Write the results and the settings that produced them."""
        directory = self.directory
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "results.json").write_text(json.dumps(results, indent=2))
        # Not the task's attributes: those hold keys, so a record of them says
        # `name=model--tags=['fr','gru']` and not the hidden size, the sparsity
        # threshold or the learning rate that key stands for.
        (directory / "manifest.json").write_text(
            json.dumps(
                {"started": self._started, **manifest.describe(self)},
                indent=2,
                default=str,
            )
        )
        return directory


class SPPTask(Task):
    """Trains a select-then-predict model over a corpus, once per seed.

    The pieces are registration keys, so the same task definition swaps its
    corpus or its model without touching code. ``preprocessor`` is optional
    only for a corpus that needs none; HateXplain has no label until one has
    run, and the loader says so rather than guessing.

    Each seed trains from scratch, restores the checkpoint that scored best on
    validation, and is evaluated on validation and test. ``faithfulness`` adds
    the terms of :mod:`pyhighlights.components.faithfulness` over the test
    split, and is off by default: they are two more columns rather than a
    correction, and a registered reproduction should report what its paper
    reports. What lands on disk is
    ``results.json`` -- every seed's metrics, plus their mean and standard
    deviation -- ``manifest.json``, and, when asked, one
    ``predictions-seed=<seed>.pkl`` per seed.
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
        add_special_tokens: bool = True,
        embeddings: str | Path | None = None,
        pretrained_tokens_only: bool = True,
        monitor: str = "val_loss",
        patience: int = 5,
        store_predictions: bool = False,
        keep_checkpoints: bool = True,
        save_weights_only: bool = False,
        faithfulness: bool = False,
        highlight_supervision: bool = False,
        highlight_loss: RegistrationKey[Loss] | None = None,
        highlight_coefficient: float = 1.0,
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
        self.add_special_tokens = add_special_tokens
        self.embeddings = Path(embeddings) if embeddings is not None else None
        self.pretrained_tokens_only = pretrained_tokens_only
        if self.embeddings is not None and pretrained_model_card is not None:
            raise ValueError(
                "a task embeds its tokens either with a pretrained model card "
                "or with a vector file, not both"
            )
        self.monitor = monitor
        self.patience = patience
        self.store_predictions = store_predictions
        # A checkpoint holds the whole model. On a transformer grid that is
        # hundreds of gigabytes of files nothing downstream reads: the task
        # restores the best one itself before scoring, and an analyzer reads
        # `results.json` and the stored predictions. Deleted after scoring
        # rather than never written -- scoring the weights training happened
        # to end on is a different experiment from scoring the best epoch.
        self.keep_checkpoints = keep_checkpoints
        # Weights without the optimizer state, which is most of a file for a
        # fine-tuned encoder. Enough to restore and score; not enough to
        # resume training, which nothing here does.
        self.save_weights_only = save_weights_only
        self.faithfulness = faithfulness
        self._embedding_matrix: th.Tensor | None = None
        self.highlight_supervision = highlight_supervision
        self.highlight_loss = highlight_loss
        self.highlight_coefficient = highlight_coefficient
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
        return load_splits(self.loader, self.preprocessor)

    def tokenizer(self, splits: Mapping[str, pd.DataFrame]) -> HighlightTokenizer:
        """A subword tokenizer when a model card is named, else a vocabulary.

        The vocabulary is fitted on ``train`` only, and its size has to match
        the backbone's ``vocab_size``: an id the embedding has no row for is a
        crash at the first batch. Naming ``embeddings`` fits it against a
        vector file instead, and the matrix that comes back is handed to the
        model, which sizes its table to it.
        """
        if self.pretrained_model_card is not None:
            return HuggingFaceTokenizer(
                self.pretrained_model_card,
                add_special_tokens=self.add_special_tokens,
            )
        # Fitted on the training split alone, and on nothing at all when a
        # corpus has none: a vocabulary that saw the evaluation text leaks it.
        training = [splits["train"]] if "train" in splits else []
        if self.embeddings is not None:
            words = {
                token
                for frame in training
                for tokens in frame["tokens"]
                for token in tokens
            }
            table, self._embedding_matrix = load_vectors(
                self.embeddings,
                tokens=words,
                pretrained_only=self.pretrained_tokens_only,
            )
            return VocabularyTokenizer(table)
        return VocabularyTokenizer(vocabulary(training, size=self.vocabulary_size))

    def loaders(self, splits: Mapping[str, pd.DataFrame]) -> Dict[str, DataLoader]:
        if self.highlight_supervision:
            self.check_supervision(splits)
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
        # Passed only when asked for, so an unsupervised run builds exactly the
        # model its key describes.
        supervision = (
            {
                "supervise_highlights": True,
                "highlight_loss": self.highlight_loss,
                "highlight_coefficient": self.highlight_coefficient,
            }
            if self.highlight_supervision
            else {}
        )
        model = Registry.from_key(
            self.model,
            expected_type=Model,
            train_metrics=self.train_metrics,
            val_metrics=self.val_metrics,
            test_metrics=self.test_metrics,
            **supervision,
        )
        # The vectors are data, so they reach the model as a tensor rather than
        # through a registration: no configuration should carry a matrix.
        if self._embedding_matrix is not None:
            model.load_embeddings(self._embedding_matrix)
        return model

    def check_supervision(self, splits: Mapping[str, pd.DataFrame]) -> None:
        """Refuse to call a run supervised when nothing supervises it.

        The collator pads unannotated positions with ``-1`` and the criterion
        skips them, so supervising a corpus annotated on test alone trains
        exactly as an unsupervised run does -- and reports itself as the
        ceiling that run was measured against. Checked where the loaders are
        built, which is the one thing every path to ``fit`` goes through.
        """
        train = splits.get("train")
        if train is None or not train["highlights"].notna().any():
            raise ValueError(
                f"{self.name}: highlight supervision needs an annotated train "
                "split, and this corpus has none"
            )

    def fit(self, seed: int, loaders: Mapping[str, DataLoader]) -> Dict[str, float]:
        """Train one model and score it, leaving its checkpoint behind."""
        seed_everything(seed=seed, workers=True)
        model = self.build_model()

        checkpoints = self.directory / f"seed={seed}"
        checkpoints.mkdir(parents=True, exist_ok=True)
        checkpoint = ModelCheckpoint(
            monitor=self.monitor,
            mode="min",
            dirpath=checkpoints,
            save_weights_only=self.save_weights_only,
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

        scores = self.score(
            trainer, model, loaders, self.directory / f"predictions-seed={seed}.pkl"
        )
        if not self.keep_checkpoints:
            self.discard_checkpoints(checkpoints)
        return scores

    def discard_checkpoints(self, directory: Path) -> None:
        """Delete this seed's checkpoints, now that they have been scored.

        What a run is read from survives: the metrics, the manifest, the
        stored predictions and, for a search, ``search.json``. The weights do
        not, so a number cannot be re-derived without training again -- which
        is the trade a grid of transformer cells makes to fit on a filesystem,
        and why this is off by default.
        """
        for checkpoint in sorted(directory.glob("*.ckpt")):
            checkpoint.unlink()

    def score(
        self,
        trainer: L.Trainer,
        model: Model,
        loaders: Mapping[str, DataLoader],
        predictions: Path,
    ) -> Dict[str, float]:
        """Score a trained model on whichever evaluation splits exist.

        ``predictions`` is where this seed's predictions go, named rather than a
        directory: they belong to the run, not to the checkpoint, and a reader
        that finds them beside a checkpoint can only say which seed produced
        them, not which run.
        """
        results: Dict[str, float] = {}
        if "val" in loaders:
            results.update(trainer.validate(model, dataloaders=loaders["val"])[0])
        if "test" in loaders:
            if self.store_predictions:
                model.enable_storing_predictions()
            results.update(trainer.test(model, dataloaders=loaders["test"])[0])
            if self.store_predictions:
                pd.to_pickle(model.predictions, predictions)
                model.flush_predictions()
                model.disable_storing_predictions()
            # After the metrics rather than beside them: the terms need the
            # predictor run against masks of their own, so they are a stage
            # over the split rather than another binding inside a test step.
            # Test only -- a validation faithfulness number selects nothing.
            if self.faithfulness:
                # Lightning moves the model back to the CPU when it tears a
                # loop down, and this stage runs outside every loop: without
                # this the terms are three CPU passes over the test split at
                # the end of a GPU run, which on a transformer is most of the
                # run's wall clock.
                model.to(trainer.strategy.root_device)
                results.update(
                    {
                        f"test_{name}": value
                        for name, value in faithfulness.evaluate(
                            model, loaders["test"]
                        ).items()
                    }
                )
        return results

    def run(self) -> Dict[str, Any]:
        # A second run of the same instance is a second result, not an
        # amendment to the first: drop the stamp so it takes its own.
        self._started = None
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


class GenSPPTask(SPPTask):
    """GenSPP: a genetic search over generators, scored like any other task.

    The corpus, the preprocessing, the metrics and the seeds are an
    :class:`SPPTask`'s. What differs is the training: no gradient reaches the
    generator, so the model is not named directly but by the
    :class:`~pyhighlights.components.models.spp.genspp.GenSPPTrainer` that
    searches for it -- one search per seed, seeded with it, since a genetic
    search over a population of two dozen is the noisiest part of the run.

    Each seed leaves behind the weights the search settled on and
    ``search.json``, the best fitness of every generation: a search that
    stopped improving in its tenth generation and one that was still climbing
    when the budget ran out report the same number otherwise.
    """

    def __init__(self, search: RegistrationKey[GenSPPTrainer], **kwargs):
        # Supervision would have to condition the population the search draws
        # from; no gradient reaches the generator, so the loss the flag adds
        # would train nothing. Left open rather than silently accepted.
        if kwargs.get("highlight_supervision"):
            raise ValueError(
                "GenSPP searches its generator rather than training it, so "
                "highlight supervision has nothing to guide"
            )
        self.search = search
        # The model key lives on the search: naming it twice is a way for the
        # two to disagree about which model was actually evolved.
        trainer = Registry.from_key(search, expected_type=GenSPPTrainer)
        super().__init__(model=trainer.model, **kwargs)

    def fit(self, seed: int, loaders: Mapping[str, DataLoader]) -> Dict[str, float]:
        if "val" not in loaders:
            raise ValueError("GenSPP scores its candidates on a validation split")

        seed_everything(seed=seed, workers=True)
        search = Registry.from_key(self.search, expected_type=GenSPPTrainer, seed=seed)
        model = search.fit(loaders["train"], loaders["val"])

        # The search builds its candidates from the model key alone, so the
        # winner arrives without metrics; they are only ever read after it.
        model.val_metrics = build_metrics(self.val_metrics)
        model.test_metrics = build_metrics(self.test_metrics)

        directory = self.directory / f"seed={seed}"
        directory.mkdir(parents=True, exist_ok=True)
        th.save({"state_dict": model.state_dict()}, directory / "best.ckpt")
        (directory / "search.json").write_text(
            json.dumps({"training_progress": search.training_progress}, indent=2)
        )
        trainer = L.Trainer(**{"default_root_dir": directory, **self.trainer_args})
        scores = self.score(
            trainer, model, loaders, self.directory / f"predictions-seed={seed}.pkl"
        )
        if not self.keep_checkpoints:
            # `search.json` stays: it is the record of how the search went,
            # not a copy of the weights.
            self.discard_checkpoints(directory)
        return scores


class ClassWeightsTask(Task):
    """Loads a corpus, weighs its classes, and writes the numbers down.

    A weighted loss needs one number per class, and where that number comes
    from decides whether a run can be repeated. Computing it inside training
    leaves it nowhere afterwards; typing it into a configuration by hand leaves
    it nowhere it can be checked. This is the third way: a run of its own,
    whose result is the weights and the counts they came from, in the same
    ``results.json`` and ``manifest.json`` every other task writes.

    So the numbers are readable by a person, persist after the process that
    computed them exits, and carry the key of the corpus and the preprocessing
    that produced them -- which is what makes them worth copying into a
    configuration, where every training run's manifest then records them.

    It trains nothing and takes no seeds.
    """

    def __init__(
        self,
        loader: RegistrationKey[HighlightLoader],
        weights: RegistrationKey[ClassWeights],
        preprocessor: RegistrationKey[Preprocessor] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.loader = loader
        self.weights = weights
        self.preprocessor = preprocessor

    def run(self) -> Dict[str, Any]:
        splits = load_splits(self.loader, self.preprocessor)
        weighting = Registry.from_key(self.weights, expected_type=ClassWeights)
        weighting.process(splits)

        results = {
            "split": weighting.split,
            "weights": weighting.weights,
            "counts": {str(label): count for label, count in weighting.counts.items()},
            # Every split, not just the weighted one: a training frequency is
            # only worth reading beside the frequencies it will be evaluated
            # against.
            "rows": {name: len(frame) for name, frame in splits.items()},
            "labels": {
                name: {
                    str(label): int(count)
                    for label, count in frame["label"].value_counts().items()
                }
                for name, frame in splits.items()
            },
        }
        logger.info("%s: class weights %s", self.name, weighting.weights)
        self.serialize(results)
        return results
