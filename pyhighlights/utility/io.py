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


def _checked_members(members: List[tarfile.TarInfo]) -> List[tarfile.TarInfo]:
    """Only files and directories, on top of the name check.

    Checking names is not enough for a tar, because a name is checked against
    the archive and extraction happens against the filesystem. A member may be
    a symlink, and ``extractall`` creates it: ``link -> ../outside`` followed by
    ``link/payload`` writes outside the staging directory through a path that
    holds no ``..`` of its own. Python 3.14 refuses that by default and every
    version this package supports does not, so the refusal is here rather than
    in an extraction filter.

    A corpus archive needs nothing else -- the released ERASER movies tar is
    2006 files and 2 directories -- so refusing the rest costs nothing and
    leaves no link semantics to reason about.
    """
    for member in members:
        if not (member.isfile() or member.isdir()):
            raise ValueError(
                f"refusing to extract archive member {member.name}: "
                "a corpus archive holds only files and directories"
            )
    return members


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
            members = source.getmembers()
            _checked_names([member.name for member in members])
            source.extractall(staging, members=_checked_members(members))
    else:
        raise ValueError(f"{archive} is neither a zip nor a tar archive")
    staging.replace(directory)
    return directory
