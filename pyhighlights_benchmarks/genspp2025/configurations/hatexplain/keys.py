"""Keys the HateXplain half of the reproduction is addressed by.

Split from the shared ones so neither corpus imports a module carrying the
other's.
"""

from pyhighlights_benchmarks.genspp2025.configurations.keys import key

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
