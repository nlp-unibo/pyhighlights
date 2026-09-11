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


TOY = key("dataset", "toy")
TOY_BACKBONE = key("backbone", "toy")
TOY_GENSPP_BACKBONE = key("backbone", "toy", "genspp")
TOY_PREDICTOR = key("predictor", "toy")
TOY_GUIDER = key("guider", "toy")
TOY_FR = key("model", "fr", "toy")
TOY_MGR = key("model", "mgr", "toy")
TOY_MCD = key("model", "mcd", "toy")
TOY_GRAT = key("model", "grat", "toy")
TOY_GENSPP = key("model", "genspp", "toy")
TOY_GENSPP_TRAINER = key("trainer", "genspp", "toy")
TOY_FR_TASK = key("task", "fr", "toy")
TOY_MGR_TASK = key("task", "mgr", "toy")
TOY_MCD_TASK = key("task", "mcd", "toy")
TOY_GRAT_TASK = key("task", "grat", "toy")
TOY_GENSPP_TASK = key("task", "genspp", "toy")
TOY_BENCHMARK = key("benchmark", "toy")

HATEXPLAIN_LENGTH_FILTER = key("preprocessor", "length", "hatexplain")
HATEXPLAIN_LABEL_MAPPER = key("preprocessor", "labels", "hatexplain")
HATEXPLAIN_AGGREGATOR = key("preprocessor", "aggregator", "hatexplain")
HATEXPLAIN_PIPELINE = key("preprocessor", "pipeline", "hatexplain")
HATEXPLAIN_SPARSITY = key("criterion", "sparsity", "hatexplain")
HATEXPLAIN_SPARSITY_LOSS = key("loss", "sparsity", "hatexplain")
HATEXPLAIN_GUIDE_LOSS = key("loss", "guide", "hatexplain")
HATEXPLAIN_JSD_LOSS = key("loss", "jsd", "hatexplain")
HATEXPLAIN_BACKBONE = key("backbone", "hatexplain")
HATEXPLAIN_GENSPP_BACKBONE = key("backbone", "hatexplain", "genspp")
HATEXPLAIN_GUIDER = key("guider", "hatexplain")
HATEXPLAIN_FR = key("model", "fr", "hatexplain")
HATEXPLAIN_MGR = key("model", "mgr", "hatexplain")
HATEXPLAIN_MCD = key("model", "mcd", "hatexplain")
HATEXPLAIN_GRAT = key("model", "grat", "hatexplain")
HATEXPLAIN_GENSPP = key("model", "genspp", "hatexplain")
HATEXPLAIN_GENSPP_TRAINER = key("trainer", "genspp", "hatexplain")
HATEXPLAIN_FR_TASK = key("task", "fr", "hatexplain")
HATEXPLAIN_MGR_TASK = key("task", "mgr", "hatexplain")
HATEXPLAIN_MCD_TASK = key("task", "mcd", "hatexplain")
HATEXPLAIN_GRAT_TASK = key("task", "grat", "hatexplain")
HATEXPLAIN_GENSPP_TASK = key("task", "genspp", "hatexplain")
HATEXPLAIN_BENCHMARK = key("benchmark", "hatexplain")

#: The paper waits thirty epochs before giving up on a five-hundred-epoch
#: budget, which is the reproduction's policy rather than the library's.
PAPER_EARLY_STOPPING = key("callback", "early_stopping")
PAPER_CHECKPOINT = key("callback", "checkpoint")
