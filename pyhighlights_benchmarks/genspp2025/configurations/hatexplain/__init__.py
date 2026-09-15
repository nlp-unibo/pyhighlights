"""HateXplain as the paper prepares it, which is not as it is distributed.

Three choices turn the corpus into the paper's benchmark, and all three change
the numbers:

* posts over thirty tokens are **dropped**, which is how the released code
  bounds its compute;
* ``offensive`` is folded into ``hatespeech`` **before** the annotators are
  counted, leaving a two-class task -- fold it afterwards and a post the three
  annotators split three ways gets a different label;
* the surviving votes and rationales are reduced by majority.

Tokens are embedded with GloVe ``twitter.27B`` at 25 dimensions, frozen.

**The vocabulary is GloVe's, not the corpus's.** The released collator is
built with ``use_pretrained_only=True``, under which it ignores the dataframe
it is handed and takes the whole of ``twitter.27B`` as its vocabulary. So no
evaluation token is unknown that GloVe covers, and that is what
``vocabulary_from="vectors"`` reproduces. Fitting on the training split
instead leaves 5.4% of validation tokens and 5.6% of test tokens embedded as
zero.

That file is a 1.4 GB download the paper expects you to fetch yourself, so it
is a path the task is given rather than a URL it fetches::

    Registry.from_key(HATEXPLAIN_FR_TASK, embeddings="glove.twitter.27B.25d.txt")

Leaving it out is refused rather than run: without the file there is no
vocabulary to take.
"""
