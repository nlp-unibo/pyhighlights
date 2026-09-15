"""Keys the Toy half of the reproduction is addressed by.

Split from the shared ones so neither corpus imports a module carrying the
other's.
"""

from pyhighlights_benchmarks.genspp2025.configurations.keys import key

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
