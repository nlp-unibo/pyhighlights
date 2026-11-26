"""Which token ids the encoders of a model read.

Configurations and components both need the answer: a condition refuses a
model whose encoders disagree, and a task reads its tokenizer from the model.
The answer comes from configurations alone, so this module imports cinnamon
and nothing that builds a component.
"""

from __future__ import annotations

from typing import Any, Set

from cinnamon.configuration import Configuration
from cinnamon.registry import RegistrationKey, Registry

__all__ = ["token_spaces"]


def token_spaces(value: Any) -> Set[str | None]:
    """The token ids the encoders under ``value`` read.

    ``value`` is a registration key, a configuration, or a list or dict of
    them, as a dependency field holds. A configuration naming
    ``pretrained_model_card`` reads the subword vocabulary of that model. A
    configuration naming ``vocabulary_size`` reads the vocabulary a task fits on its
    corpus, which is ``None`` here. Dependencies are walked, so the backbone
    of a guider counts as a backbone of the model holding it.

    Every encoder of one model reads the same ids, so a model has at most one
    token space. A task reads the model card from it rather than naming one.
    """
    if isinstance(value, RegistrationKey):
        value = Registry.retrieve_configuration(value)
    if isinstance(value, (list, tuple)):
        return set().union(*(token_spaces(item) for item in value))
    if isinstance(value, dict):
        return set().union(*(token_spaces(item) for item in value.values()))
    if not isinstance(value, Configuration):
        return set()
    if "pretrained_model_card" in value.fields:
        return {value.pretrained_model_card}
    spaces: Set[str | None] = {None} if "vocabulary_size" in value.fields else set()
    for dependency in value.dependencies.values():
        spaces |= token_spaces(dependency)
    return spaces
