"""The synthetic corpus: three hidden patterns over a twenty-character string.

Tokens are characters, and the corpus turns out to use twenty-four of them --
no ``i`` and no ``x``. With the unknown and padding id that is a vocabulary of
twenty-five, which is where the released baselines' ``embedding_dim=25`` comes
from: their table is not a learned projection at all but a one-hot matrix as
wide as the vocabulary, so the number is the vocabulary's size rather than a
hyperparameter. The genetic half declares twenty-six dimensions for the same
twenty-four characters, leaving two columns always zero.

``one_hot_embeddings`` on the task is what supplies that matrix. A frozen
*random* table, which is what this reproduction had before, is a different
corpus to learn from: its rows have norm five and reach a cosine of 0.58 with
each other, where one-hot rows are orthonormal.
"""

#: Twenty-four characters plus the unknown and padding id.
VOCABULARY_SIZE = 25
