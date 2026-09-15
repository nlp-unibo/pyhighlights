"""The synthetic corpus: three hidden patterns over a twenty-character string.

Tokens are characters, and the corpus turns out to use twenty-four of them --
no ``i`` and no ``x``. With the unknown and padding id that is a vocabulary of
twenty-five ids, and a one-hot code over them needs **twenty-four** columns:
id zero is the zero row and carries none.

**The release uses two different widths and neither is that number.** The
baselines build their table as ``one_hot(arange(0, 25), num_classes=25)`` with
row zero zeroed, so column zero is never set. The genetic half hardcodes
``OneHotEmbedder(vocab_size=26)`` (``TOY_TOKEN_EMBEDDING_DIM = 26``), leaving
two columns never set. The two halves therefore disagree about the width of
the same corpus.

The disagreement costs nothing but weights. A column that is always zero
contributes nothing to a GRU and receives no gradient, and two backbones at
widths 26 and 24 -- same weights on the live columns, same ids in -- encode
bitwise identically. What the extra columns buy is 24 and 48 input weights
that never move. So this reproduction takes the width the alphabet actually
needs and notes the release's numbers rather than carrying them.

Which column is the dead one also differs, and matters as little:
:func:`~pyhighlights.utility.embeddings.one_hot_table` puts id ``j`` at column
``j - 1`` and leaves the last column unset, where the release's ``one_hot``
puts it at column ``j`` and leaves the first. A permutation of the input
columns is absorbed by the projection reading them, so it is a different
matrix and the same model -- but released weights could not be loaded into
this table without one.

``one_hot_embeddings`` on the task is what supplies the matrix. A frozen
*random* table, which is what this reproduction had before, is a different
corpus to learn from: its rows have norm five and reach a cosine of 0.58 with
each other, where one-hot rows are orthonormal.
"""

#: Twenty-four characters plus the unknown and padding id.
VOCABULARY_SIZE = 25

#: One column per character, which is one fewer than there are ids: the
#: unknown and padding id is the zero row. The released baselines declare 25
#: and the genetic half 26; both leave columns that are never set.
EMBEDDING_DIM = VOCABULARY_SIZE - 1
