"""HateXplain as the paper prepares it, which is not as it is distributed.

Three choices turn the corpus into the paper's benchmark, and all three change
the numbers:

* posts over thirty tokens are **dropped**, which is how the released code
  bounds its compute;
* ``offensive`` is folded into ``hatespeech`` **before** the annotators are
  counted, leaving a two-class task -- fold it afterwards and a post the three
  annotators split three ways gets a different label;
* the surviving votes and rationales are reduced by majority.

Tokens are embedded with GloVe ``twitter.27B`` at 25 dimensions, frozen, and
the vocabulary is restricted to what the release covers. That file is a
1.4 GB download the paper expects you to fetch yourself, so it is a path the
task is given rather than a URL it fetches::

    Registry.from_key(HATEXPLAIN_FR_TASK, embeddings="glove.twitter.27B.25d.txt")
"""

#: GloVe twitter covers what it covers; the table is replaced on load, so this
#: is a placeholder rather than a number anybody has to get right.
VOCABULARY_SIZE = 2
