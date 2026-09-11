import json
from pathlib import Path

import pytest
from cinnamon.registry import Registry

import pyhighlights
from pyhighlights.components.loaders import (
    BeerLoader,
    ERASERLoader,
    HateXplainLoader,
    HotelLoader,
    MoviesLoader,
    R2ALoader,
    ToyLoader,
    to_examples,
)
from pyhighlights.configurations.keys import BEER, HATEXPLAIN, HOTEL, MOVIES, TOY
from tests.corpora import UNPINNED, eraser, hatexplain, r2a


def hotel(tmp_path: Path, **kwargs) -> HotelLoader:
    return HotelLoader(
        url=r2a(tmp_path), directory=tmp_path / "cache", **UNPINNED, **kwargs
    )


def test_r2a_downloads_once_and_parses_the_standard_columns(tmp_path):
    source = hotel(tmp_path)
    splits = source.load()

    assert set(splits) == {"train", "val", "test"}
    assert list(splits["train"].columns) == [
        "sample_id",
        "text",
        "tokens",
        "label",
        "highlights",
    ]
    # The splits come back as distributed: the annotated rows are still in
    # training, and nothing has been dropped.
    assert len(splits["train"]) == 3
    assert splits["train"]["highlights"].isna().all()

    test = splits["test"]
    assert test["tokens"].iloc[1] == ["close", "to", "the", "station"]
    assert test["highlights"].iloc[1] == [0, 0, 0, 1]
    assert list(test["sample_id"]) == [0, 1]

    # The archive is fetched once; a second load reuses the extracted copy.
    (source.directory / source.archive_name).unlink()
    assert source.read()["test"].equals(test)


def test_beer_and_hotel_accept_only_their_own_aspects(tmp_path):
    assert (
        BeerLoader(url=r2a(tmp_path), directory=tmp_path / "b", **UNPINNED).task
        == "beer0"
    )
    assert hotel(tmp_path).task == "hotel_Location"

    with pytest.raises(ValueError, match="task must be one of"):
        BeerLoader(task="hotel_Location")
    with pytest.raises(ValueError, match="task must be one of"):
        HotelLoader(task="beer0")
    # The shared parser still takes either.
    assert R2ALoader(task="beer1").task == "beer1"


def test_frames_convert_to_highlight_examples(tmp_path):
    datasets = hotel(tmp_path).datasets()
    example = datasets["test"][1]

    assert len(datasets["train"]) == 3
    assert example.highlights == (0, 0, 0, 1)
    assert datasets["train"][0].highlights is None


def test_misaligned_rationales_are_rejected(tmp_path):
    path = tmp_path / "broken.train"
    path.write_text(
        "task\tlabel\ttext\trationale\tpred_att\n"
        "hotel_Location\t1\ttwo tokens\t0 1 1\t0.1\n"
    )

    with pytest.raises(ValueError, match="misaligned"):
        R2ALoader.read_file(path)


def test_continuous_labels_are_rejected(tmp_path):
    path = tmp_path / "source.train"
    path.write_text("task\tlabel\ttext\nbeer0\t0.3\tno taste at all\n")

    with pytest.raises(ValueError, match="continuous labels"):
        R2ALoader.read_file(path)


def test_toy_corpus_marks_its_trigger_and_repeats_for_a_seed():
    loader = ToyLoader(sizes={"train": 8, "test": 4}, length=6, seed=3)
    splits = loader.load()

    assert list(splits) == ["train", "test"]
    assert len(splits["train"]) == 8

    row = splits["train"].iloc[0]
    marked = [token for token, flag in zip(row.tokens, row.highlights) if flag]
    assert " ".join(marked) == "a great film"
    assert len(row.highlights) == len(row.tokens) == 9
    assert sorted(splits["train"]["label"].unique()) == [0, 1]

    assert ToyLoader(seed=3).load()["train"].equals(ToyLoader(seed=3).load()["train"])
    assert (
        not ToyLoader(seed=4).load()["train"].equals(ToyLoader(seed=3).load()["train"])
    )

    with pytest.raises(ValueError, match="one trigger per class"):
        ToyLoader(triggers=["only one"])


def test_hatexplain_keeps_every_annotator_judgement(tmp_path):
    splits = HateXplainLoader(**hatexplain(tmp_path)).load()

    # Nothing is aggregated and nothing is dropped: the tie post survives, and
    # the label and highlights wait for a preprocessor.
    assert [len(frame) for frame in splits.values()] == [2, 1, 1]
    assert splits["train"]["label"].isna().all()
    assert splits["train"]["highlights"].isna().all()
    assert splits["train"]["annotator_labels"].iloc[0] == [
        "hatespeech",
        "hatespeech",
        "offensive",
    ]
    assert splits["train"]["annotator_highlights"].iloc[0] == [[1, 1, 0], [1, 0, 0]]


def test_unaggregated_splits_refuse_to_become_examples(tmp_path):
    loader = HateXplainLoader(**hatexplain(tmp_path))

    with pytest.raises(ValueError, match="AnnotationAggregator"):
        loader.datasets()


def test_eraser_turns_evidence_spans_into_highlights(tmp_path):
    splits = MoviesLoader(**eraser(tmp_path)).load()

    train = splits["train"].iloc[0]
    assert train.tokens == ["a", "truly", "awful", "film", "not", "worth", "it"]
    assert train.highlights == [0, 1, 1, 0, 0, 1, 1]
    assert train.label == 0

    # An evidence-free row is annotated with nothing marked, not left unlabelled.
    assert splits["val"]["highlights"].iloc[0] == [0, 0, 0]
    assert splits["test"]["highlights"].iloc[0] == [0, 1, 0]
    assert splits["test"]["label"].iloc[0] == 1


def test_eraser_refuses_query_based_tasks_and_unknown_ones():
    with pytest.raises(ValueError, match="pairs a query with its document"):
        ERASERLoader(task="multirc")
    with pytest.raises(ValueError, match="task must be one of"):
        ERASERLoader(task="imdb")


def test_eraser_rejects_spans_outside_the_document(tmp_path):
    settings = eraser(tmp_path)
    root = tmp_path / "cache" / "eraser" / "movies" / "movies"
    loader = MoviesLoader(**settings)
    loader.download()
    (root / "train.jsonl").write_text(
        json.dumps(
            {
                "annotation_id": "d2.txt",
                "classification": "POS",
                "docids": None,
                "evidences": [[{"docid": "d2.txt", "start_token": 2, "end_token": 9}]],
            }
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="evidence span outside"):
        MoviesLoader(**settings).load()


def test_registered_loaders_build(tmp_path):
    Registry.build(directory=Path(pyhighlights.__file__).parent)

    assert isinstance(Registry.from_key(TOY, sizes={"train": 4}), ToyLoader)
    assert isinstance(
        Registry.from_key(HATEXPLAIN, **hatexplain(tmp_path / "hx")), HateXplainLoader
    )
    assert isinstance(
        Registry.from_key(MOVIES, **eraser(tmp_path / "er")), MoviesLoader
    )

    archive = r2a(tmp_path)
    beer = Registry.from_key(BEER, url=archive, directory=str(tmp_path / "b"))
    hotel_loader = Registry.from_key(HOTEL, url=archive, directory=str(tmp_path / "h"))
    assert isinstance(beer, BeerLoader) and beer.task == "beer0"
    assert isinstance(hotel_loader, HotelLoader)
    assert hotel_loader.task == "hotel_Location"


def test_to_examples_needs_the_standard_columns(tmp_path):
    frame = hotel(tmp_path).load()["train"].drop(columns=["label"])

    with pytest.raises(KeyError):
        to_examples(frame)
