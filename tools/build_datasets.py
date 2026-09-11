"""Build the upload-ready pyhighlights dataset artifacts.

Three of the four artifacts are *split manifests*: retained upstream row
indices, the source URL and checksum, counts, the repair policy and the
citation. They carry no review text, so republishing them raises no licence
question about the corpora they index. The fourth, the GenSPP toy corpus, is
the complete dataset and is the one that needs the authors' sign-off.

The build is deterministic: every zip entry gets a fixed timestamp and mode,
every JSON payload is key-sorted, and the entries are written in name order.
Two runs from clean produce byte-identical zips, which is what lets
``checksums.sha256`` be quoted in the Zenodo record.

Run from the build directory -- the one holding ``sources`` and ``dist`` --
or name it with ``--work-dir``. ``--selfcheck`` asserts determinism on a cheap
artifact without downloading the 162 MB R2A archive, and asserts that the
movies manifest reconstructs its own splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

import pyhighlights
from pyhighlights.components.loaders import (
    BEER_TASKS,
    HOTEL_TASKS,
    R2A_SPLITS,
    R2A_URL,
    BeerLoader,
    HotelLoader,
    MoviesLoader,
)
from pyhighlights.components.preprocessors import PRIORITY, remove_leakage
from pyhighlights.utility.io import download

# The build directory is wherever you run this from, not wherever the script
# lives: the script is versioned in the library checkout and the 690 MB of
# pinned sources are not.
SOURCES = Path.cwd() / "sources"
DIST = Path.cwd() / "dist"


def configure(work_dir: Path) -> None:
    """Point the build at ``work_dir``, which holds ``sources`` and ``dist``."""
    global SOURCES, DIST
    SOURCES = work_dir / "sources"
    DIST = work_dir / "dist"


VERSION = "v1"

# Pinned upstream sources. The R2A digest is the one the library's loaders and
# the previous build both verified against.
R2A_SHA256 = "23fcb4cac883ec1de86d83a7747294d7fdae10061d3803fd4c34c930e66f25de"
GENSPP_TOY_URL = (
    "https://raw.githubusercontent.com/nlp-unibo/gen-spp/main/genetic/data/"
    "toy_dataset.pkl"
)

R2A_CITATION = (
    "Bao, Y., Chang, S., Yu, M., & Barzilay, R. (2018). Deriving Machine "
    "Attention from Human Rationales. EMNLP 2018."
)
ERASER_CITATION = (
    "DeYoung, J., Jain, S., Rajani, N. F., Lehman, E., Xiong, C., Socher, R., "
    "& Wallace, B. C. (2020). ERASER: A Benchmark to Evaluate Rationalized "
    "NLP Models. ACL 2020."
)
GENSPP_CITATION = (
    "Ruggeri, F., & Signorelli, G. (2025). Interlocking-free Selective "
    "Rationalization Through Genetic-based Learning. Proceedings of the 63rd "
    "Annual Meeting of the Association for Computational Linguistics "
    "(Volume 1: Long Papers), 1175-1191. "
    "https://doi.org/10.18653/v1/2025.acl-long.59 "
    "Reference implementation: https://github.com/nlp-unibo/gen-spp"
)

# One licence across every record, which is also Zenodo's default. A manifest
# holds no part of the corpus it points at, so CC0 was defensible for it, but
# uniformity is worth more than the one attribution it would have saved: a
# reader comparing four records sees one set of terms, and the repair policy
# and the indices are authored work that attribution is a fair condition on.
MANIFEST_LICENSE = "CC-BY-4.0"
GENSPP_TOY_LICENSE = "CC-BY-4.0"

REPAIR = {
    "policy": "pyhighlights.components.preprocessors.remove_leakage",
    "priority": list(PRIORITY),
    "key": "text",
    "normalize_keys": True,
    "note": (
        "Splits are walked in priority order; each keeps only rows no earlier "
        "split claimed and no earlier row of its own repeated. The annotated "
        "split is kept whole, so training and validation are what give rows "
        "up. Indices below index rows of the upstream file, counting from 0."
    ),
}


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_digest(path: Path) -> str:
    return digest(path.read_bytes())


def write_zip(target: Path, entries: List[Tuple[str, bytes]]) -> Path:
    """Write a zip whose bytes depend only on its entries' names and payloads."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, payload in sorted(entries, key=lambda entry: entry[0]):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, payload)
    return target


def as_json(payload: dict) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def split_manifest(splits: Dict[str, pd.DataFrame]) -> dict:
    """Retained upstream indices and counts, after the leakage repair.

    ``remove_leakage`` renumbers ``sample_id`` but leaves other columns alone,
    so the upstream row index rides along in a column of its own rather than
    being recomputed here -- the manifest cannot drift from what the library
    actually does to a corpus.
    """
    tagged = {
        name: frame.assign(upstream_index=frame.index.astype(int))
        for name, frame in splits.items()
    }
    repaired = remove_leakage(
        tagged,
        priority=REPAIR["priority"],
        key=REPAIR["key"],
        normalize_keys=REPAIR["normalize_keys"],
    )
    return {
        "counts": {
            name: {
                "distributed": int(len(splits[name])),
                "retained": int(len(repaired[name])),
                "removed": int(len(splits[name]) - len(repaired[name])),
            }
            for name in splits
        },
        "label_counts": {
            name: {
                str(label): int(count)
                for label, count in sorted(
                    repaired[name]["label"].value_counts().items()
                )
            }
            for name in splits
        },
        "annotated": {
            name: bool(repaired[name]["highlights"].notna().all()) for name in splits
        },
        "retained_upstream_indices": {
            name: [int(index) for index in repaired[name]["upstream_index"]]
            for name in splits
        },
    }


def r2a_artifact(
    family: str, loader_class, tasks
) -> Tuple[str, List[Tuple[str, bytes]]]:
    name = f"pyhighlights-r2a-{family}-splits-{VERSION}"
    entries: List[Tuple[str, bytes]] = []
    summary = {}
    for task in tasks:
        loader = loader_class(
            task=task, directory=SOURCES, sha256=R2A_SHA256, archive_name="r2a.zip"
        )
        manifest = {
            "artifact": name,
            "task": task,
            "loader": f"{loader_class.__module__}.{loader_class.__qualname__}",
            "pyhighlights_version": pyhighlights.__version__,
            "source": {
                "url": R2A_URL,
                "sha256": R2A_SHA256,
                "files": {
                    split: f"data/{pattern.format(task=task)}"
                    for split, pattern in R2A_SPLITS.items()
                },
            },
            "repair": REPAIR,
            "citation": R2A_CITATION,
            "license": MANIFEST_LICENSE,
            "contents": "split manifest only -- no review text is redistributed",
            **split_manifest(loader.load()),
        }
        entries.append((f"{task}/manifest.json", as_json(manifest)))
        summary[task] = manifest["counts"]
    entries.append(
        (
            "README.md",
            readme(name, R2A_URL, R2A_SHA256, R2A_CITATION, summary),
        )
    )
    return name, entries


def movies_artifact() -> Tuple[str, List[Tuple[str, bytes]]]:
    name = f"pyhighlights-eraser-movies-splits-{VERSION}"
    loader = MoviesLoader(directory=SOURCES)
    url = loader.url.format(task=loader.task)
    # Fetch before digesting: the archive does not exist on a clean build, and
    # a manifest that records ``null`` on the first run and a digest on the
    # second is not a reproducible artifact.
    loader.download()
    archive = SOURCES / f"eraser-{loader.task}.tar.gz"
    manifest = {
        "artifact": name,
        "task": loader.task,
        "loader": "pyhighlights.components.loaders.MoviesLoader",
        "pyhighlights_version": pyhighlights.__version__,
        "source": {
            "url": url,
            "sha256": file_digest(archive),
            "files": dict(loader.splits),
        },
        "repair": REPAIR,
        "citation": ERASER_CITATION,
        "license": MANIFEST_LICENSE,
        "contents": "split manifest only -- no document text is redistributed",
        **split_manifest(loader.load()),
    }
    entries = [
        ("movies/manifest.json", as_json(manifest)),
        (
            "README.md",
            readme(
                name,
                url,
                manifest["source"]["sha256"],
                ERASER_CITATION,
                {loader.task: manifest["counts"]},
            ),
        ),
    ]
    return name, entries


def genspp_toy_artifact() -> Tuple[str, List[Tuple[str, bytes]]]:
    """The complete toy corpus -- the one artifact that redistributes data."""
    name = f"pyhighlights-genspp-toy-{VERSION}"
    pickle = download(GENSPP_TOY_URL, SOURCES / "genspp-toy_dataset.pkl")
    payload = pickle.read_bytes()
    corpus = pd.read_pickle(pickle)
    manifest = {
        "artifact": name,
        "loader": "pyhighlights_benchmarks.genspp2025.corpora.GenSPPToyLoader",
        "pyhighlights_version": pyhighlights.__version__,
        "source": {
            "url": GENSPP_TOY_URL,
            "sha256": digest(payload),
            "file": "toy_dataset.pkl",
        },
        "contents": "the complete dataset, as released",
        "rows": int(len(corpus)),
        "label_counts": {
            str(label): int(count)
            for label, count in sorted(corpus["label"].value_counts().items())
        },
        "tokens": "characters; the vocabulary is the lowercase alphabet",
        "split_scheme": {
            "note": (
                "Splits are not stored: the loader derives them from the "
                "released baselines' scheme, so the artifact is the corpus "
                "and the split is code."
            ),
            "train_ratio": 0.8,
            "val_ratio": 0.2,
            "split_seed": 15000,
        },
        "citation": GENSPP_CITATION,
        "license": GENSPP_TOY_LICENSE,
    }
    entries = [
        ("toy_dataset.pkl", payload),
        ("manifest.json", as_json(manifest)),
        (
            "README.md",
            readme(
                name,
                GENSPP_TOY_URL,
                manifest["source"]["sha256"],
                GENSPP_CITATION,
                {"toy": {"rows": manifest["rows"]}},
            ),
        ),
    ]
    return name, entries


def readme(name, url, sha256, citation, summary) -> bytes:
    manifest_only = "genspp-toy" not in name
    license_name = MANIFEST_LICENSE if manifest_only else GENSPP_TOY_LICENSE
    body = [
        f"# {name}",
        "",
        "Published for [pyhighlights](https://github.com/nlp-unibo/pyhighlights)",
        f"{pyhighlights.__version__}, so that a reproduction reads the same rows"
        " every time.",
        "",
        "## What is in here",
        "",
    ]
    if manifest_only:
        body += [
            "**Split manifests only.** Each `manifest.json` lists the upstream",
            "row indices each split retains after the zero-leakage repair, with",
            "counts, label counts and the repair policy that produced them. No",
            "review or document text is redistributed: fetch the corpus from",
            "the source below, then index it with these manifests.",
        ]
    else:
        body += [
            "**The complete corpus**, `toy_dataset.pkl`, as released, plus a",
            "`manifest.json` recording its checksum, row count and the split",
            "scheme the loader derives.",
        ]
    body += [
        "",
        "## Licence",
        "",
        f"This artifact is licensed **{license_name}**.",
        "",
    ]
    body += (
        [
            "These terms cover the manifest -- the retained row indices, the",
            "counts and the repair policy -- and nothing else. The corpus you",
            "fetch from the source below carries whatever terms its own",
            "release carries; cite it as given under Citation.",
        ]
        if manifest_only
        else [
            "The corpus is released under these terms by both authors of the",
            "paper that produced it. Attribution means the citation given",
            "below.",
        ]
    )
    body += [
        "",
        "## Source",
        "",
        f"- URL: `{url}`",
        f"- SHA-256: `{sha256}`",
        "",
        "## Counts",
        "",
        "```json",
        json.dumps(summary, indent=2, sort_keys=True),
        "```",
        "",
        "## Citation",
        "",
        citation,
        "",
    ]
    return ("\n".join(body)).encode()


def build(dist: Path, include_r2a: bool = True) -> dict:
    artifacts = []
    if include_r2a:
        artifacts.append(r2a_artifact("beer", BeerLoader, BEER_TASKS))
        artifacts.append(r2a_artifact("hotel", HotelLoader, HOTEL_TASKS))
        artifacts.append(movies_artifact())
    artifacts.append(genspp_toy_artifact())

    report = {"pyhighlights_version": pyhighlights.__version__, "artifacts": {}}
    lines = []
    if include_r2a:
        # Uploaded beside the zips: Zenodo serves each file of a record at its
        # own URL, and ``GenSPPToyLoader`` reads a pickle rather than an
        # archive, so the loader's default URL has to name this file directly.
        loose = dist / "toy_dataset.pkl"
        loose.parent.mkdir(parents=True, exist_ok=True)
        loose.write_bytes((SOURCES / "genspp-toy_dataset.pkl").read_bytes())
        lines.append(f"{file_digest(loose)}  {loose.name}")
        report["loose_files"] = {
            loose.name: {"bytes": loose.stat().st_size, "sha256": file_digest(loose)}
        }
    for name, entries in artifacts:
        target = write_zip(dist / f"{name}.zip", entries)
        checksum = file_digest(target)
        report["artifacts"][name] = {
            "file": target.name,
            "bytes": target.stat().st_size,
            "sha256": checksum,
            "entries": sorted(entry for entry, _ in entries),
        }
        lines.append(f"{checksum}  {target.name}")
    (dist / "checksums.sha256").write_bytes(("\n".join(sorted(lines)) + "\n").encode())
    (dist / "build-report.json").write_bytes(as_json(report))
    return report


def check_applicable(manifest: dict, splits: Dict[str, pd.DataFrame]) -> None:
    """A manifest has to reconstruct the repaired splits it describes.

    Indices that cannot be applied are a record of nothing, so this is the
    check worth keeping: take the corpus as distributed, select the manifest's
    rows, and require the result to equal what ``remove_leakage`` produces.
    """
    repaired = remove_leakage(
        splits,
        priority=REPAIR["priority"],
        key=REPAIR["key"],
        normalize_keys=REPAIR["normalize_keys"],
    )
    for name, indices in manifest["retained_upstream_indices"].items():
        rebuilt = splits[name].loc[indices]
        assert list(rebuilt["text"]) == list(repaired[name]["text"]), (
            f"{manifest['artifact']} {name}: indices do not rebuild the split"
        )
        assert len(indices) == manifest["counts"][name]["retained"]


def selfcheck() -> None:
    """Two builds of the same artifact must agree byte for byte."""
    first = build(DIST / "_selfcheck-a", include_r2a=False)
    second = build(DIST / "_selfcheck-b", include_r2a=False)
    assert first == second, "the build is not deterministic"
    for name, entry in first["artifacts"].items():
        a = (DIST / "_selfcheck-a" / entry["file"]).read_bytes()
        b = (DIST / "_selfcheck-b" / entry["file"]).read_bytes()
        assert a == b, f"{name} differs between two builds"

    # Movies is the cheap manifest artifact -- 3.8 MB rather than R2A's 162 --
    # so it is the one the round-trip check runs against.
    loader = MoviesLoader(directory=SOURCES)
    _, entries = movies_artifact()
    manifest = json.loads(dict(entries)["movies/manifest.json"])
    check_applicable(manifest, loader.load())
    print("selfcheck: deterministic, and the movies manifest rebuilds its splits")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path.cwd(),
        help="directory holding sources/ and dist/ (default: the current one)",
    )
    parser.add_argument("--selfcheck", action="store_true")
    parser.add_argument(
        "--skip-r2a",
        action="store_true",
        help="build only the GenSPP toy artifact (no 162 MB download)",
    )
    arguments = parser.parse_args()
    configure(arguments.work_dir.resolve())
    if arguments.selfcheck:
        selfcheck()
        return 0
    report = build(DIST, include_r2a=not arguments.skip_r2a)
    for name, entry in sorted(report["artifacts"].items()):
        print(f"{entry['bytes']:>9,} B  {entry['sha256'][:12]}  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
