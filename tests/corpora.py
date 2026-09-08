"""Miniature corpora the loader and preprocessing tests share."""

import json
import tarfile
import zipfile
from pathlib import Path

LABELLED = "task\tlabel\ttext\n"
ANNOTATED = "task\tlabel\ttext\trationale\tpred_att\n"


def r2a(directory: Path) -> str:
    """A miniature R2A archive: the test split leaks fully into train."""
    leaked = "the location is central but noisy"
    files = {
        "data/oracle/hotel_Location.train": LABELLED
        + f"hotel_Location\t1\t{leaked}\n"
        + "hotel_Location\t0\ta room with no windows\n"
        + "hotel_Location\t1\tclose to the station\n",
        "data/oracle/hotel_Location.dev": LABELLED
        + "hotel_Location\t0\tfar from everything\n",
        # Whitespace differs from the training row on purpose.
        "data/target/hotel_Location.train": ANNOTATED
        + "hotel_Location\t1\tThe  location is central   but noisy\t0 1 0 0 0 0\t0.1\n"
        + "hotel_Location\t1\tclose to the station\t0 0 0 1\t0.2\n",
    }
    path = directory / "mini.zip"
    with zipfile.ZipFile(path, "w") as target:
        for name, content in files.items():
            target.writestr(name, content)
    return path.as_uri()


def annotator(label: str, index: int) -> dict:
    return {"label": label, "annotator_id": index, "target": ["None"]}


def hatexplain(directory: Path) -> dict:
    """Posts covering a majority label, a tie, a normal post and a duplicate."""
    directory.mkdir(parents=True, exist_ok=True)
    posts = {
        "p1": {
            "post_id": "p1",
            "annotators": [
                annotator("hatespeech", 1),
                annotator("hatespeech", 2),
                annotator("offensive", 3),
            ],
            "rationales": [[1, 1, 0], [1, 0, 0]],
            "post_tokens": ["burn", "them", "all"],
        },
        "p2": {
            "post_id": "p2",
            "annotators": [
                annotator("hatespeech", 1),
                annotator("offensive", 2),
                annotator("normal", 3),
            ],
            "rationales": [[0, 1]],
            "post_tokens": ["who", "cares"],
        },
        "p3": {
            "post_id": "p3",
            "annotators": [annotator("normal", i) for i in (1, 2, 3)],
            "rationales": [],
            "post_tokens": ["nice", "day"],
        },
        # Same text as p1: id-based splits do not stop text leaking.
        "p4": {
            "post_id": "p4",
            "annotators": [
                annotator("hatespeech", 1),
                annotator("hatespeech", 2),
                annotator("normal", 3),
            ],
            "rationales": [[1, 1, 0], [1, 1, 0], [0, 1, 0]],
            "post_tokens": ["burn", "them", "all"],
        },
    }
    divisions = {"train": ["p1", "p2"], "val": ["p3"], "test": ["p4"]}
    (directory / "posts.json").write_text(json.dumps(posts))
    (directory / "divisions.json").write_text(json.dumps(divisions))
    return {
        "url": (directory / "posts.json").as_uri(),
        "divisions_url": (directory / "divisions.json").as_uri(),
        "directory": directory / "cache",
    }


def eraser(directory: Path) -> dict:
    """A miniature single-document ERASER task."""
    directory.mkdir(parents=True, exist_ok=True)
    documents = {"d1.txt": "a truly awful film\nnot worth it", "d2.txt": "a fine film"}
    rows = {
        "train.jsonl": [
            {
                "annotation_id": "d1.txt",
                "classification": "NEG",
                "docids": None,
                "evidences": [
                    [
                        {
                            "docid": "d1.txt",
                            "start_token": 1,
                            "end_token": 3,
                            "text": "truly awful",
                        }
                    ],
                    [
                        {
                            "docid": "d1.txt",
                            "start_token": 5,
                            "end_token": 7,
                            "text": "worth it",
                        }
                    ],
                ],
            }
        ],
        "val.jsonl": [
            {
                "annotation_id": "d2.txt",
                "classification": "POS",
                "docids": ["d2.txt"],
                "evidences": [],
            }
        ],
        "test.jsonl": [
            {
                "annotation_id": "d2.txt",
                "classification": "POS",
                "docids": None,
                "evidences": [
                    [
                        {
                            "docid": "d2.txt",
                            "start_token": 1,
                            "end_token": 2,
                            "text": "fine",
                        }
                    ]
                ],
            }
        ],
    }
    staging = directory / "movies"
    (staging / "docs").mkdir(parents=True)
    for name, text in documents.items():
        (staging / "docs" / name).write_text(text)
    for name, lines in rows.items():
        (staging / name).write_text(
            "\n".join(json.dumps(line) for line in lines) + "\n"
        )

    archive = directory / "movies.tar.gz"
    with tarfile.open(archive, "w:gz") as target:
        target.add(staging, arcname="movies")
    return {"url": archive.as_uri(), "directory": directory / "cache"}
