"""Download and archive helpers shared by the corpus loaders."""

from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import List


def cache_directory() -> Path:
    """Shared download cache, overridable with ``PYHIGHLIGHTS_CACHE``."""
    default = Path.home() / ".cache" / "pyhighlights"
    return Path(os.environ.get("PYHIGHLIGHTS_CACHE") or default)


def download(url: str, target: Path, sha256: str | None = None) -> Path:
    """Fetch ``url`` into ``target`` unless already there; verify if asked."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        partial = target.with_name(target.name + ".part")
        urllib.request.urlretrieve(url, partial)
        partial.replace(target)

    if sha256 is not None:
        digest = hashlib.sha256()
        with target.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        if digest.hexdigest() != sha256:
            raise ValueError(f"{target} does not match the expected sha256")
    return target


def _checked_names(names: List[str]) -> List[str]:
    for name in names:
        parts = Path(name).parts
        if Path(name).is_absolute() or ".." in parts:
            raise ValueError(f"refusing to extract unsafe archive path {name}")
    return names


def extract(archive: Path, directory: Path) -> Path:
    """Unpack ``archive`` into ``directory`` once; return ``directory``."""
    if directory.exists():
        return directory

    staging = directory.with_name(directory.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as source:
            source.extractall(staging, members=_checked_names(source.namelist()))
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as source:
            _checked_names(source.getnames())
            source.extractall(staging)
    else:
        raise ValueError(f"{archive} is neither a zip nor a tar archive")
    staging.replace(directory)
    return directory
