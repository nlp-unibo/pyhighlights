"""Keys the GenSPP reproduction is addressed by.

Its own namespace, so the paper's HateXplain -- filtered to short posts, folded
to two classes -- never stands in for the corpus as distributed.
"""

from cinnamon.registry import RegistrationKey

NAMESPACE = "genspp2025"

#: The five seeds every reported number averages over.
SEEDS = [2023, 15451, 1337, 2001, 2080]


def key(name: str, *tags: str) -> RegistrationKey:
    return RegistrationKey(name=name, tags=set(tags), namespace=NAMESPACE)


#: The paper waits thirty epochs before giving up on a five-hundred-epoch
#: budget, which is the reproduction's policy rather than the library's.
PAPER_EARLY_STOPPING = key("callback", "early_stopping")
PAPER_CHECKPOINT = key("callback", "checkpoint")
