"""Values that components and configurations share.

A configuration reads its defaults from here rather than from a component
module. Importing any component module imports torch, and building the registry
should not, because a registration binds its component by import path so that
nothing is built or imported until a component is requested. This module
therefore imports nothing.
"""

__all__ = [
    "BEER_TASKS",
    "ERASER_SHA256",
    "ERASER_URL",
    "HATEXPLAIN_COMMIT",
    "HATEXPLAIN_DIVISIONS_SHA256",
    "HATEXPLAIN_DIVISIONS_URL",
    "HATEXPLAIN_LABELS",
    "HATEXPLAIN_SHA256",
    "HATEXPLAIN_URL",
    "HOTEL_TASKS",
    "PREDICTIONS",
    "PRIORITY",
    "R2A_SHA256",
    "R2A_URL",
]

R2A_URL = "https://people.csail.mit.edu/yujia/files/r2a/data.zip"
#: The release the split manifests on Zenodo were built against. Pinned so a
#: reproduction fails loudly on a changed upstream rather than training on it.
R2A_SHA256 = "23fcb4cac883ec1de86d83a7747294d7fdae10061d3803fd4c34c930e66f25de"
BEER_TASKS = ("beer0", "beer1", "beer2")
HOTEL_TASKS = ("hotel_Location", "hotel_Service", "hotel_Cleanliness")

#: The commit both HateXplain files are read at. A branch name is not a
#: version: the same key would name different rows after an upstream push, and
#: a run made before it could not be told from a run made after. The benchmark
#: publishes no digest of its own, so :data:`HATEXPLAIN_SHA256` and
#: :data:`HATEXPLAIN_DIVISIONS_SHA256` were computed against this commit, which
#: is what ``master`` resolved to as of 2026-09-15, byte for byte.
HATEXPLAIN_COMMIT = "01d742279dac941981f53806154481c0e15ee686"
HATEXPLAIN_URL = (
    "https://raw.githubusercontent.com/hate-alert/HateXplain/"
    f"{HATEXPLAIN_COMMIT}/Data/dataset.json"
)
#: Digest of the 12256170-byte ``dataset.json`` that commit holds.
HATEXPLAIN_SHA256 = "63bb3340fee0ec469b09690d04cb68f7c187787dd8b83807f071892c084967fb"
HATEXPLAIN_DIVISIONS_URL = (
    "https://raw.githubusercontent.com/hate-alert/HateXplain/"
    f"{HATEXPLAIN_COMMIT}/Data/post_id_divisions.json"
)
#: Digest of the official split map. Separate from :data:`HATEXPLAIN_SHA256`
#: because they are separate downloads: one can change without the other.
HATEXPLAIN_DIVISIONS_SHA256 = (
    "c2fb0d89862e7897b11ea3e9380753f15a793482b4b70ad0532dfb1212212835"
)
HATEXPLAIN_LABELS = ("hatespeech", "normal", "offensive")

ERASER_URL = "https://www.eraserbenchmark.com/zipped/{task}.tar.gz"
#: The archive the ``movies`` split manifest was built against. The benchmark
#: publishes no digest of its own. One constant serves because ERASER support
#: is one task; a second would want this keyed by task.
ERASER_SHA256 = "66e18d4e6c9df9e9f5544572b0bfe92a39673f74ecbfc3859b46cedb2f5b2dee"

#: Priority runs from the split that must stay intact to the one that can
#: afford to lose rows. The annotated split comes first: it is the only one
#: carrying highlights, so it is the one worth protecting.
PRIORITY = ("test", "val", "train")

#: The predictions one seed left behind. A glob rather than a name: a run
#: stores one file per seed, and every analyzer reads all of them.
PREDICTIONS = "predictions-seed=*.pkl"
