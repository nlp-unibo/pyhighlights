# pyhighlights

Research library for highlight-based explainable AI models.

## GRU folded rationalization

```python
from pathlib import Path

import pyhighlights
from cinnamon.registry import Registry
from pyhighlights.configurations.spp import (
    GRU_FR,
    GRU_GRAT,
    GRU_MCD,
    GRU_MGR,
    TRANSFORMER_FR,
)

Registry.build(directory=Path(pyhighlights.__file__).parent)
fr = Registry.from_key(GRU_FR)
mgr = Registry.from_key(GRU_MGR)
mcd = Registry.from_key(GRU_MCD)
grat = Registry.from_key(GRU_GRAT)
transformer_fr = Registry.from_key(TRANSFORMER_FR)  # needs pyhighlights[transformers]
```

GRU and Transformer implementations conform to same backbone interface; model
classes contain rationalization logic only.

## Highlight data

```python
from torch.utils.data import DataLoader
from pyhighlights.components import (
    HighlightCollator,
    HighlightDataset,
    HighlightExample,
    VocabularyTokenizer,
)

examples = HighlightDataset([
    HighlightExample(0, ["great", "stay"], label=1, highlights=[1, 0]),
    HighlightExample(1, ["bad"], label=0),  # unlabeled highlights become -1
])
collator = HighlightCollator(VocabularyTokenizer({"great": 1, "stay": 2, "bad": 3}))
batch = next(iter(DataLoader(examples, batch_size=2, collate_fn=collator)))
```

`HuggingFaceTokenizer` expands word highlights across subtokens and requires
`pyhighlights[transformers]`.
