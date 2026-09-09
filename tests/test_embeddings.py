import pytest
import torch as th

from pyhighlights.components.models.spp.implementations import GRUBackbone
from pyhighlights.utility.embeddings import load_vectors


def vectors_file(tmp_path, lines=("cat 1 0", "dog 0 1", "fish 1 1")):
    path = tmp_path / "vectors.txt"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_vectors_are_read_into_a_vocabulary_reserving_the_unknown_id(tmp_path):
    vocabulary, matrix = load_vectors(vectors_file(tmp_path))

    assert vocabulary == {"cat": 1, "dog": 2, "fish": 3}
    assert matrix.shape == (4, 2)
    # Row 0 is the unknown and padding id, and stays zero.
    assert th.equal(matrix[0], th.zeros(2))
    assert th.equal(matrix[1], th.tensor([1.0, 0.0]))


def test_only_the_requested_tokens_are_kept(tmp_path):
    vocabulary, matrix = load_vectors(vectors_file(tmp_path), tokens=["dog", "cat"])

    # Ids follow the file, not the request: the matrix is built as it is read.
    assert vocabulary == {"cat": 1, "dog": 2}
    assert matrix.shape == (3, 2)


def test_a_token_without_a_vector_is_dropped_or_randomised(tmp_path):
    path = vectors_file(tmp_path)
    words = ["cat", "unicorn"]

    dropped, matrix = load_vectors(path, tokens=words, pretrained_only=True)
    assert dropped == {"cat": 1}
    assert matrix.shape == (2, 2)

    kept, matrix = load_vectors(
        path, tokens=words, pretrained_only=False, generator=th.Generator()
    )
    assert set(kept) == {"cat", "unicorn"}
    # The released vector is untouched; the invented one is not zero.
    assert th.equal(matrix[kept["cat"]], th.tensor([1.0, 0.0]))
    assert matrix[kept["unicorn"]].any()


def test_released_file_quirks_are_read_through(tmp_path):
    """A byte order mark, a word2vec header and a blank line."""
    path = tmp_path / "vectors.txt"
    path.write_text("3 2\ncat 1 0\n\ndog 0 1\n", encoding="utf-8-sig")

    vocabulary, matrix = load_vectors(path)

    assert vocabulary == {"cat": 1, "dog": 2}
    assert matrix.shape == (3, 2)


def test_an_unusable_file_says_so(tmp_path):
    with pytest.raises(ValueError, match="covers none of the requested tokens"):
        load_vectors(vectors_file(tmp_path), tokens=["unicorn"])

    ragged = vectors_file(tmp_path, lines=("cat 1 0", "dog 0 1 1"))
    with pytest.raises(ValueError, match="different widths"):
        load_vectors(ragged)


def test_a_backbone_sizes_its_table_to_the_matrix_and_keeps_it_frozen():
    backbone = GRUBackbone(
        vocab_size=3, embedding_dim=2, hidden_size=4, freeze_embeddings=True
    )
    matrix = th.arange(10, dtype=th.float32).reshape(5, 2)

    backbone.load_embeddings(matrix)

    # The pretrained vocabulary is as wide as the file covers, which the
    # configuration could not have guessed.
    assert backbone.embedding.num_embeddings == 5
    assert th.equal(backbone.embedding.weight, matrix)
    assert backbone.embedding.weight.requires_grad is False

    trainable = GRUBackbone(vocab_size=3, embedding_dim=2, hidden_size=4)
    trainable.load_embeddings(matrix)
    assert trainable.embedding.weight.requires_grad is True

    # The width the GRU was built for is what a matrix has to match.
    with pytest.raises(ValueError, match="3-dimensional.*expects 2"):
        backbone.load_embeddings(th.zeros(5, 3))
