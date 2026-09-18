"""What a run cost to produce, beside what it scored.

A table of F1 says which model is better and nothing about what it takes to
get there. These are the other half: how long a seed ran, how long one
inference batch and one inference pass take, how much memory the run reached,
how many parameters the model carries, and -- for a genetic search -- how many
models were trained at once to produce the one that got scored.

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
``cost_memory_per_run_mb``
    The peak divided by the workers that were resident in it.

The workers counted are the ones that had something to do: a pool of eight
scoring a population of four runs four at a time.
"""

from __future__ import annotations

import resource
import sys
import time
from typing import Any, Dict, List

import lightning as L
import torch as th

__all__ = ["InferenceTimer", "Meter", "parameters", "peak_memory"]


def parameters(model: th.nn.Module) -> int:
    """Every parameter the model carries, trained or frozen.

    Frozen included: a frozen transformer is memory and compute at inference
    however little it learns, and inference is what the other columns time.
    """
    return sum(parameter.numel() for parameter in model.parameters())


def peak_memory() -> float:
    """Megabytes at the high-water mark, on the device the run used.

    CUDA reports the run's own peak, since :class:`Meter` resets the counter
    when it starts. The CPU figure is the **process**'s high-water mark, which
    only ever rises: a second seed in the same process inherits the first's
    peak rather than measuring its own. That is what the operating system
    offers, and it is still the honest ceiling for a run of one seed.
    """
    if th.cuda.is_available() and th.cuda.max_memory_allocated():
        return th.cuda.max_memory_allocated() / 1e6
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Kilobytes on Linux, bytes on macOS, for the same field.
    return peak / 1e6 if sys.platform == "darwin" else peak / 1024


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
            "cost_memory_mb": self.peak,
            "cost_memory_per_run_mb": self.peak / self.concurrency,
            "cost_parameters": float(parameters(model)),
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
