# pyhighlights

[![Tests](https://github.com/nlp-unibo/pyhighlights/actions/workflows/ci.yml/badge.svg)](https://github.com/nlp-unibo/pyhighlights/actions/workflows/ci.yml)
[![Documentation](https://github.com/nlp-unibo/pyhighlights/actions/workflows/docs.yml/badge.svg)](https://nlp-unibo.github.io/pyhighlights/)
[![PyPI](https://img.shields.io/pypi/v/pyhighlights)](https://pypi.org/project/pyhighlights/)

Select-then-predict models for highlight-based explainable AI research.

Most explainability methods answer *why did the model say that?* after the
fact, and nothing forces the answer to be true. A select-then-predict model
rearranges the pipeline so the question does not arise: a **selector** picks a
subset of the input, and a **predictor** sees only that subset. The selected
subset — the **highlight** — is not a story about the prediction, it is the
input to it. A highlight that omits what mattered produces a worse prediction,
which is measurable rather than arguable.

[Documentation](https://nlp-unibo.github.io/pyhighlights/) ·
[Tutorial](https://nlp-unibo.github.io/pyhighlights/quickstart.html) ·
[Contributing](CONTRIBUTING.md)

## Installation

```bash
pip install pyhighlights
```

Transformer backbones are an extra, so a GRU run does not pull in
`transformers`:

```bash
pip install "pyhighlights[transformers]"
```

## One key, one experiment

A task is one experiment start to finish — a corpus, its preprocessing, a
model, its metrics, and a list of seeds:

```python
from pathlib import Path

import pyhighlights
from cinnamon.registry import Registry
from pyhighlights.components.analyzers import MetricsAnalyzer, PredictionAnalyzer
from pyhighlights.configurations.keys import TOY_TASK

Registry.build(directory=Path(pyhighlights.__file__).parent)

task = Registry.from_key(
    TOY_TASK,
    save_path="results",
    seeds=[0, 1],
    store_predictions=True,
)
task.run()
```

Each seed trains from scratch, restores the checkpoint that scored best on
validation, and is scored on validation and test. Seeds are a list rather than
a number because one run of a select-then-predict model says very little: the
selector is trained through a discrete choice, and the spread across seeds is
part of the result.

What lands on disk:

```
results/toy/2026-09-09T17-55-22/
├── results.json               # every seed's metrics, and their summary
├── manifest.json              # the key, the overrides, the whole
│                              #   configuration tree, and the versions
├── predictions-seed=0.pkl
├── predictions-seed=1.pkl
└── seed=0/epoch=0-step=8.ckpt
```

A run never overwrites an earlier one: two runs of a task are two results to
compare. The manifest names the key and the arguments the run was launched
with, and resolves every nested key into the numbers behind it — so the file
states the hidden size and the learning rate rather than the name of the place
they came from.

Read the results back as frames, so the same analyzer serves a notebook, a
test and a LaTeX table:

```python
MetricsAnalyzer(directory="results", metrics=["accuracy", "highlight_f1"]).run()
```

```
task                 run  seeds          accuracy      highlight_f1
 toy 2026-09-09T17-55-22      2 0.4062 +/- 0.0312 0.0000 +/- 0.0000
```

A stored prediction is token ids and masks. `PredictionAnalyzer` rebuilds the
corpus from the key in the manifest and joins on `sample_id`, so a row says
which words the model kept:

```python
PredictionAnalyzer(directory="results").analyze()[
    ["seed", "sample_id", "label", "predicted", "rationale"]
]
```

## What is in it

**Architectures**, each a different answer to selection being a discrete
choice inside a differentiable model. All seven are registered for both a GRU
and a Transformer backbone; the algorithms never mention either, because a
backbone is anything implementing `encode` / `pool` / `output_size`.

| | |
|---|---|
| **FR** | Selector and predictor fold onto one encoder, so they cannot drift apart. Liu et al., NeurIPS 2022 |
| **MGR** | Several generators, one shared predictor, so no single degenerate generator sets the equilibrium. Liu et al., ACL 2023 |
| **MCD** | Trained against selected-input and full-input predictions; agreement means the highlight d-separates the label. Liu et al., NeurIPS 2023 |
| **MRD** | The predictor reads the complement and the full input, and the generator maximizes the discrepancy between them, so a spurious feature degenerates to noise. Liu et al., NeurIPS 2024 |
| **DR** | The predictor trains at the selector's rate scaled by how much of the input the selection kept, which restrains its Lipschitz constant. Liu et al., KDD 2023 |
| **G-RAT** | A pretrained attention classifier guides the selection and matches its distribution. Hu and Yu, AAAI 2024 |
| **GenSPP** | No gradient reaches the generator — a genetic search scores each candidate by training a fresh predictor. Ruggeri and Signorelli, ACL 2025 |

**Corpora**, each loaded as its authors distributed it: R2A `beer` and
`hotel` (three aspects each), ERASER `movies`, `hatexplain`, and a synthetic
`toy` for smoke tests. Cleaning is a preprocessor a study names, not something
a loader does, so two studies over one corpus can prepare it differently.

**Metrics** per corpus, by output classes and by highlights, plus faithfulness
— sufficiency and comprehensiveness — over the test split.

**Reproductions** live in `pyhighlights_benchmarks`, beside the library rather
than inside it, so nothing here carries one paper's values. GenSPP (ACL 2025)
is the first: two corpora against all five architectures.

## Where it is going

[Roadmap](https://nlp-unibo.github.io/pyhighlights/roadmap.html) — Zenodo
dataset artifacts so a reproduction does not depend on a URL somebody else
controls, a study on unannotated legal text, per-corpus sparsity targets, and
the rest of the highlight-metric suite.

pyhighlights ships **tools**, not an experiment. Which metrics to log, which
aspect of Beer to train on, which sparsity target to aim at — all of that
depends on a study, so all of it stays a configuration the study writes.
