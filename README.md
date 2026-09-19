<div align="center">

# pyhighlights

[![Tests](https://github.com/nlp-unibo/pyhighlights/actions/workflows/ci.yml/badge.svg)](https://github.com/nlp-unibo/pyhighlights/actions/workflows/ci.yml)
[![Documentation](https://github.com/nlp-unibo/pyhighlights/actions/workflows/docs.yml/badge.svg)](https://nlp.unibo.it/pyhighlights/)
[![PyPI](https://img.shields.io/pypi/v/pyhighlights)](https://pypi.org/project/pyhighlights/)

</div>

Models for highlight-based explainable AI research.

A highlight-based model commits to a subset of the document, the **highlight**,
and predicts from that subset alone.
The highlight is therefore not a story about the prediction, it is the input to
it.
A highlight that omits what mattered produces a worse prediction, which is
measurable rather than arguable.

The methods are many and the benchmarks are many, and no standard way of
running them exists.
Every paper brings its own corpora, its own splits and its own metrics, so two
published numbers rarely answer the same question.
Reproducing a result is hard and comparing two methods fairly is harder.

pyhighlights exists to remove that obstacle.
It runs any model it ships on any corpus it ships, under one configuration
system and one set of metrics.
A comparison is then a change of key rather than a rewrite.

The library ships the select-then-predict family, where a **selector** picks
the highlight and a **predictor** reads only what the selector kept.
Other families, dual-head architectures among them, come next.
The corpora, the metrics and the experiment machinery belong to the library
rather than to a family.
Adding a family is therefore an architecture and its configurations, and
nothing else.

[Documentation](https://nlp.unibo.it/pyhighlights/) ·
[Tutorial](https://nlp.unibo.it/pyhighlights/tutorials/quickstart.html) ·
[Contributing](CONTRIBUTING.md)

## Installation

The library requires Python 3.10 or later.

```bash
pip install pyhighlights
```

Transformer backbones are an extra, so a GRU run does not pull in
`transformers`:

```bash
pip install "pyhighlights[transformers]"
```

## Quickstart

A task is one experiment start to finish: a corpus, its preprocessing, a model,
its metrics, and a list of seeds.
Write the script below to a file and run it from any directory you can write
to.
The results land beneath the `save_path` it names.

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

```bash
python quickstart.py
```

Each seed trains from scratch, restores the checkpoint that scored best on
validation, and is scored on validation and test.
Seeds are a list rather than a number because one run says very little.
The selector is trained through a discrete choice, which makes the
optimization unstable.
The spread across seeds is therefore part of the result.

What lands on disk:

```
results/toy/2026-09-19T16-41-53/
├── results.json               # every seed's metrics, and their summary
├── manifest.json              # the key, the overrides, the whole
│                              #   configuration tree, and the versions
├── predictions-seed=0.pkl
├── predictions-seed=1.pkl
├── seed=0/epoch=1-step=16.ckpt
└── seed=0/lightning_logs/version_0/metrics.csv
```

A run never overwrites an earlier one, so two runs of a task are two results to
compare.
The manifest names the key and the arguments the run was launched with.
It also resolves every nested key, so it states the hidden size and the
learning rate themselves rather than the name they came from.

Read the results back as frames, so the same analyzer serves a notebook, a test
and a LaTeX table:

```python
MetricsAnalyzer(directory="results", metrics=["accuracy", "highlight_f1"]).run()
```

```
task                 run  seeds          accuracy      highlight_f1
 toy 2026-09-19T16-41-53      2 0.7188 +/- 0.2812 0.3489 +/- 0.3063
```

`TOY_TASK` is a smoke test rather than a result.
It trains FR on a GRU backbone for two epochs on CPU over 64 generated
documents, which checks that a run holds together and teaches nothing else.
Still, the spread is worth reading: two seeds of one configuration disagree by
more than a quarter of the highlight score.

A stored prediction is token ids and masks.
`PredictionAnalyzer` rebuilds the corpus from the key in the manifest and joins
it on `sample_id`, so a row says which words the model kept:

```python
PredictionAnalyzer(directory="results").analyze()[
    ["seed", "sample_id", "label", "predicted", "selected_text"]
]
```

Both analyzers report the same frame, and they differ in what they do with it:
`run` prints it and returns it, while `analyze` only returns it.

## Architectures

Selection is a discrete choice inside a differentiable model, and each
architecture answers that differently.
All eight are registered for a GRU and for a Transformer backbone.
No algorithm mentions either, since a backbone is anything implementing
`encode`, `pool` and `output_size`.

| Name | Idea | Reference |
|---|---|---|
| **FR** | Selector and predictor fold onto one encoder, so they cannot drift apart. | Liu et al., NeurIPS 2022 |
| **MGR** | Several generators, one shared predictor, so no single degenerate generator sets the equilibrium. | Liu et al., ACL 2023 |
| **MCD** | Trained against selected-document and full-document predictions, where agreement means the highlight d-separates the label. | Liu et al., NeurIPS 2023 |
| **MRD** | The predictor reads the complement and the full document, and the generator maximizes the discrepancy between them, so a spurious feature degenerates to noise. | Liu et al., NeurIPS 2024 |
| **DAR** | A second predictor, trained on the full document and then frozen, has to read the highlight too, so a selection drifting from the document costs the generator. | Liu et al., ICDE 2024 |
| **DR** | The predictor trains at the selector's rate scaled by how much of the document the selection kept, which restrains its Lipschitz constant. | Liu et al., KDD 2023 |
| **G-RAT** | A pretrained attention classifier guides the selection and matches its distribution. | Hu and Yu, AAAI 2024 |
| **GenSPP** | No gradient reaches the generator, and a genetic search scores each candidate by training a fresh predictor. | Ruggeri and Signorelli, ACL 2025 |

## Corpora

Three of the five corpora leak as distributed: annotated evaluation rows also
appear in training.
Every Hotel aspect leaks its whole annotated split, and a highlight score
published on those splits is measured on seen data.
The table below reports the repaired splits, which is what `LeakageRemover`
produces and what `LeakageDetector().check()` accepts.
The distributed splits stay reachable, since a reproduction of published
numbers needs them.
The [datasets page](https://nlp.unibo.it/pyhighlights/reference/datasets.html)
gives the per-aspect counts and the digests.

Beer and Hotel carry three aspects each, and the rows below are `beer0` and
`hotel_Location`.
HateXplain is aggregated before the repair, so its three annotators per
document become one label and one highlight.
That aggregation drops no row.
The toy corpus is generated by `ToyLoader`, whose triggers, document length,
alphabet and split sizes are all parameters.
The same loader reads a published corpus when given a `url` and a digest, which
is how the released GenSPP corpus is read
([10.5281/zenodo.22711448](https://doi.org/10.5281/zenodo.22711448)).

| Corpus | Source | Domain | Classes | Train / validation / test | Median tokens | Highlight annotation | Leakage repair |
|---|---|---|---|---|---|---|---|
| **beer** | R2A archive, Bao et al., ICML 2018 | Beer reviews, one aspect rating | 2 | 27,973 / 6,388 / 200 | 125 / 123 / 115 | Test only, 17.7% of tokens | 4,303 train and 4 validation rows dropped |
| **hotel** | R2A archive, Bao et al., ICML 2018 | Hotel reviews, one aspect rating | 2 | 12,546 / 1,767 / 200 | 145 / 146 / 148 | Test only, 10.1% of tokens | 1,926 train and 45 validation rows dropped |
| **movies** | ERASER, DeYoung et al., ACL 2020 | Film reviews, sentiment | 2 | 1,599 / 200 / 199 | 730 / 740 / 710 | Every split, 9.3% / 7.2% / 31.4% of tokens | 1 duplicate train row dropped |
| **hatexplain** | Mathew et al., AAAI 2021 | Social media posts | 3 | 15,348 / 1,921 / 1,923 | 21 / 21 / 20 | Every split, 10.2% / 10.5% / 9.8% of tokens | 35 train, 1 validation and 1 test row dropped |
| **toy** | Generated, or read from a release | Character patterns | 2 | 64 / 16 / 16 | 20 / 20 / 20 | Every split, 10.0% of tokens | None, nothing leaks |

## Metrics

Classification metrics are registered per class count.
Highlight metrics score only the positions a corpus annotated, so an
unannotated position counts as nothing rather than as a negative.

| Metric | What it measures |
|---|---|
| `accuracy` | Share of documents whose class the predictor got right. |
| `f1` | F1 over the classes, macro-averaged. |
| `class_f1` | F1 of the class named by `pos_label`, for a corpus too skewed for an average to be informative. |
| `highlight_f1` | Token-level F1 of the selection against the annotated highlight. |
| `highlight_precision` | Of the positions the selection marked, the share the corpus annotates. |
| `highlight_recall` | Of the positions the corpus annotates, the share the selection marked. |
| `highlight_iou` | Intersection over union of the two token sets. |
| `selection_rate` | Share of a document the selector kept, averaged over documents. |
| `selection_size` | Tokens the selector kept, averaged over documents. |
| `selection_spans` | Contiguous runs the selection falls into, averaged over documents. |
| `sufficiency` | `p(y \| x) - p(y \| h)`, so lower is better. Whether the highlight carries the signal on its own. |
| `comprehensiveness` | `p(y \| x) - p(y \| x \ h)`, so higher is better. Whether anything the class rests on was left outside the highlight. |

Sufficiency and comprehensiveness are the faithfulness pair of DeYoung et al.,
ACL 2020, computed over the test split.
They are not two views of one number.
A model can highlight three words that suffice while ten others would have
sufficed too, which is sufficient and not comprehensive.

## Reproductions

A reproduction lives in a repository of its own.

| Study | Paper | Original implementation | Reproduction | Year |
|---|---|---|---|---|
| **GenSPP** | [Interlocking-free Selective Rationalization Through Genetic-based Learning](https://aclanthology.org/2025.acl-long.59/) | [nlp-unibo/gen-spp](https://github.com/nlp-unibo/gen-spp) | [nlp-unibo/pyhighlights-genspp2025](https://github.com/nlp-unibo/pyhighlights-genspp2025) | 2025 |

The GenSPP reproduction runs two corpora against the five architectures that
paper compares, which are FR, MCD, MGR, G-RAT and GenSPP, rather than against
all eight the library ships.

## Contact

pyhighlights ships tools rather than an experiment.
Which metrics to log, which aspect of Beer to train on and which sparsity
target to aim at all depend on a study.
Each stays a configuration the study writes.

Questions about a component or a result are best raised as an
[issue](https://github.com/nlp-unibo/pyhighlights/issues).
For anything else, write to Federico Ruggeri, federico.ruggeri6@unibo.it.
