"""What a run cost to produce, beside what it scored.

A table of F1 says which model is better and nothing about what it takes to
get there. These are the other half: how long a seed ran, how long one
inference batch and one inference pass take, how much memory the run reached
(in mebibytes), how many parameters the model carries and how many of them
gradient descent moves, and -- for a genetic search -- how many models were
trained at once to produce the one that got scored.

Every column is prefixed ``cost_``, so
:class:`~pyhighlights.components.analyzers.MetricsAnalyzer` reports them with
``split="cost"`` and leaves them out of the ``test_`` table a paper quotes.

**Concurrency is what makes these comparable.** A baseline trains one model
per seed; GenSPP trains thousands, several at a time, and reports the winner.
Wall clock alone would say a search is as cheap as the hours it happened to
take on the machine that ran it. So a run records how many models it trained
(``cost_models``) and how many of them were in flight at once
(``cost_concurrency``), and derives the per-model figures from both:

``cost_runtime_per_run_s``
    ``runtime * concurrency / models``. Workers running at once multiply the
    work done in a second, so the product is worker-seconds and dividing by
    the models trained gives what one of them cost. A baseline, at one model
    and one worker, reports its own wall clock.

The workers counted are the ones that had something to do: a pool of eight
scoring a population of four runs four at a time.

**There is no per-model memory column**, deliberately. Most of what a run
holds is the interpreter, torch and the corpus, resident before the first
candidate exists -- measured at 521 MiB with nothing training. Dividing the
peak by the workers would report less than that, which is not what any one
model costs. ``cost_memory_mib`` is the ceiling a run needs, which is the
question a machine is sized by.

A search that scores its candidates in processes is read the same way: the
figure covers this process and the largest of its children, since the
children are the same model on the same data and the largest is therefore
what one candidate costs.
"""

from __future__ import annotations

import resource
import sys
import time
from typing import Any, Dict, List

import lightning as L
import torch as th

__all__ = ["InferenceTimer", "Meter", "parameters", "peak_memory"]


#: Bytes in a mebibyte. Every memory column is in ``MiB`` -- what ``nvidia-smi``
#: and every process monitor print -- rather than in decimal megabytes, which
#: would read five percent larger for the same allocation.
MIB = 1024**2


def parameters(model: th.nn.Module, trainable: bool | None = None) -> int:
    """How many parameters the model carries.

    ``trainable`` selects: ``True`` counts what gradient descent moves,
    ``False`` what it does not, and left out counts both. All three are worth
    reporting -- a frozen encoder is memory and compute at inference however
    little it learns, and a model that freezes most of itself is a different
    proposition to train than one that does not.

    Counted on the model as it was **scored**, which is the state a reader of
    the table gets if they load the checkpoint: a component frozen partway
    through training counts as frozen, however many epochs moved it first,
    and so does a generator a genetic search settled on rather than descended
    to. Trainable here means what gradient descent moves.
    """
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if trainable is None or parameter.requires_grad is trainable
    )


def peak_memory() -> float:
    """Mebibytes at the high-water mark, on the device the run used.

    CUDA reports the run's own peak, since :class:`Meter` resets the counter
    when it starts. The CPU figure is the **process**'s high-water mark, which
    only ever rises: a second seed in the same process inherits the first's
    peak rather than measuring its own. That is what the operating system
    offers, and it is still the honest ceiling for a run of one seed.
    """
    if th.cuda.is_available() and th.cuda.max_memory_allocated():
        return th.cuda.max_memory_allocated() / MIB
    # Children as well as this process: a genetic search scores its candidates
    # in processes of their own, and `RUSAGE_SELF` would report the parent
    # waiting on them. `ru_maxrss` over children is the largest any one of them
    # reached rather than their sum, which is the right figure here anyway --
    # they are the same model on the same data, so the largest is what one
    # costs, and the ceiling is what a machine is sized by.
    peak = max(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
    )
    # Kibibytes on Linux, bytes on macOS, for the same field.
    return peak / MIB if sys.platform == "darwin" else peak / 1024


class Meter:
    """Times a seed and reads what it peaked at.

    ``models`` is how many models the seed trained -- one for a baseline, a
    search's whole population for GenSPP -- and ``concurrency`` how many of
    them ran at once. Both are settable after construction, because a search
    only knows how many generations it ran once it has stopped.
    """

    def __init__(self, concurrency: int = 1, models: int = 1):
        if concurrency < 1 or models < 1:
            raise ValueError("a run trains at least one model, on at least one worker")
        self.concurrency = concurrency
        self.models = models
        self.runtime = 0.0
        self.peak = 0.0
        self._started = 0.0

    def start(self) -> "Meter":
        if th.cuda.is_available():
            th.cuda.reset_peak_memory_stats()
        self._started = time.perf_counter()
        return self

    def stop(self) -> "Meter":
        self.runtime = time.perf_counter() - self._started
        self.peak = peak_memory()
        return self

    # A seed is trained in one place and scored in another, with a checkpoint
    # restored in between, so the two ends are called rather than wrapped.
    __enter__ = start

    def __exit__(self, *exception: Any) -> None:
        self.stop()

    def columns(self, model: th.nn.Module) -> Dict[str, float]:
        """What the seed cost, as columns of ``results.json``."""
        return {
            "cost_runtime_s": self.runtime,
            "cost_runtime_per_run_s": self.runtime * self.concurrency / self.models,
            "cost_memory_mib": self.peak,
            "cost_parameters": float(parameters(model)),
            "cost_trainable_parameters": float(parameters(model, trainable=True)),
            "cost_frozen_parameters": float(parameters(model, trainable=False)),
            "cost_concurrency": float(self.concurrency),
            "cost_models": float(self.models),
        }


class InferenceTimer(L.Callback):
    """Times the test pass: the whole of it, and each batch of it.

    Test rather than validation, because it is the pass the reported numbers
    come from and it runs once per seed on a model that has stopped training.
    The epoch figure covers what a caller waits for, loading the batches
    included; the batch figure covers the forward passes alone, which is what
    a second model on the same corpus is compared against.
    """

    def __init__(self) -> None:
        self.batches: List[float] = []
        self.epoch = 0.0
        self._batch_started = 0.0
        self._epoch_started = 0.0

    def _now(self) -> float:
        # A CUDA kernel is queued rather than run, so a timestamp taken
        # without this measures how fast Python got to the next line.
        if th.cuda.is_available():
            th.cuda.synchronize()
        return time.perf_counter()

    def on_test_epoch_start(
        self, trainer: L.Trainer, module: L.LightningModule
    ) -> None:
        self.batches = []
        self._epoch_started = self._now()

    def on_test_batch_start(self, *arguments: Any, **keywords: Any) -> None:
        self._batch_started = self._now()

    def on_test_batch_end(self, *arguments: Any, **keywords: Any) -> None:
        self.batches.append(self._now() - self._batch_started)

    def on_test_epoch_end(self, trainer: L.Trainer, module: L.LightningModule) -> None:
        self.epoch = self._now() - self._epoch_started

    def columns(self) -> Dict[str, float]:
        """What inference cost, or nothing at all if no test pass ran."""
        if not self.batches:
            return {}
        return {
            "cost_inference_epoch_s": self.epoch,
            "cost_inference_batch_s": sum(self.batches) / len(self.batches),
        }
