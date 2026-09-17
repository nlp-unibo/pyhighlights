"""Diagnostics: what the pipeline held, stage by stage, for one bounded run.

A run either finishes or raises, and nothing in between is visible. When a
number comes out wrong there is no record of what the pipeline actually held:
the frames a loader parsed, the splits a preprocessing step returned, the
batch a collator assembled, the states a backbone produced, the mask a
selector proposed, the input the predictor was handed, or the terms a loss
summed. Every one of those is otherwise reconstructed by hand, against a model
built a second time.

So the stages report themselves, through the standard library's ``logging``
under one logger. Nothing is written unless something asks for it:
:func:`active` is a level check, and a call site that finds it false costs the
check and the call. :class:`~pyhighlights.components.tasks.SPPTask` turns it on
for a run it has bounded, writing into that run's own directory beside
``results.json``.

Reading the record is the point rather than keeping it. What it answers that
nothing else does: whether the word axis and the subtoken axis agree on every
batch, whether a dropped word is absent from the predictor's input on the
backbone in use, how often the empty-selection repair fires, and whether every
loss and metric binding finds its fields on every split.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch as th

__all__ = ["active", "describe", "logger", "record", "writing"]

#: One logger for every stage. A caller that wants the record on a console
#: rather than in a file adds a handler to this and sets its level.
logger = logging.getLogger("pyhighlights.diagnostics")

#: What a run's record is called inside the run's own directory.
FILENAME = "diagnostics.log"


def active() -> bool:
    """Whether anything is listening, which is what a call site checks.

    A level check rather than a flag, so the record follows the logger the
    same way every other message in this library does, and a caller that
    configured logging themselves is not overridden.
    """
    return logger.isEnabledFor(logging.DEBUG)


def describe(value: Any) -> str:
    """One line for one thing the pipeline held.

    A tensor reports the shape, the dtype, the device, how many entries are
    not finite and the range they cover. That is enough to see a mask that is
    neither zero nor one, a selection rate of 1.0 at the first batch, or a
    ``nan`` inside a pooled state -- none of which a metric shows. A frame
    reports its rows, its columns and how many rows carry an annotation.
    Anything else reports itself, shortened, since a stage is free to name a
    number or a string beside its tensors.
    """
    if isinstance(value, th.Tensor):
        if value.numel():
            finite = th.isfinite(value)
            spread = (
                f"[{value[finite].min():.4g}, {value[finite].max():.4g}]"
                if bool(finite.any())
                else "[]"
            )
            unfinite = int((~finite).sum())
        else:
            spread, unfinite = "[]", 0
        return (
            f"tensor{tuple(value.shape)} {value.dtype} {value.device} "
            f"range={spread} non-finite={unfinite}"
        )
    # By duck typing rather than by importing pandas for an isinstance: a
    # frame is the only thing here carrying both `columns` and `shape`.
    columns = getattr(value, "columns", None)
    if columns is not None and hasattr(value, "shape"):
        annotated = ""
        if "highlights" in columns:
            marked = sum(1 for item in value["highlights"] if item is not None)
            annotated = f" annotated={marked}"
        return f"frame rows={value.shape[0]} columns={list(columns)}{annotated}"
    text = repr(value)
    return text if len(text) <= 120 else text[:117] + "..."


def record(stage: str, /, **values: Any) -> None:
    """Report what one stage held, under the name that stage goes by.

    Formatting happens only when something is listening: every value here is
    a tensor the forward pass is holding anyway, and describing one costs a
    reduction over it -- measured at 0.2 microseconds per call while nothing
    listens against 0.1 millisecond per tensor while something does, which is
    the whole reason a diagnosed run has to be a bounded one.

    ``stage`` is positional-only because the values are named by whatever the
    caller is reporting: a corpus whose splits include one called ``stage``
    would otherwise crash the run inside the call meant to explain it."""
    if not active():
        return
    for name, value in values.items():
        logger.debug("%s: %s = %s", stage, name, describe(value))


@contextmanager
def writing(directory: Path) -> Iterator[Path]:
    """Send the record to ``directory`` for the length of one run.

    The handler and the level are put back afterwards, so a task that
    diagnosed one run leaves the logger as it found it and a second task in
    the same process is not still writing into the first one's directory.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / FILENAME
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield path
    finally:
        logger.setLevel(level)
        logger.removeHandler(handler)
        handler.close()
