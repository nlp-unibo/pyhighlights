"""The synthetic corpus: three hidden patterns over a twenty-character string.

Tokens are characters, so the vocabulary is the alphabet and the embedding
table is 27 rows -- twenty-six letters and the unknown id. The released
baselines freeze that table without pretraining it, which makes it a fixed
random projection rather than something the model can learn to lean on.
"""

#: Twenty-six letters plus the unknown id.
VOCABULARY_SIZE = 27
