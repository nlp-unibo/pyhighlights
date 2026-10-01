"""Download and archive helpers shared by the corpus loaders."""

from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import urllib.request
import uuid
import zipfile
from pathlib import Path
from typing import List


def cache_directory() -> Path:
    """Shared download cache, overridable with ``PYHIGHLIGHTS_CACHE``."""
    default = Path.home() / ".cache" / "pyhighlights"
    return Path(os.environ.get("PYHIGHLIGHTS_CACHE") or default)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, target: Path, sha256: str | None = None) -> Path:
    """Fetch ``url`` into ``target`` unless already there; verify if asked.

    A download that fails leaves no corpus behind. The bytes land in a
    temporary file beside the target, are checked against ``sha256``, and are
    renamed onto the target only then, and a rename is atomic. An interrupted
    or corrupted fetch is therefore a missing file the next call retries,
    never a truncated one a loader would parse.

    Each call writes its own temporary file, so several processes may fetch
    the same target at once: under data parallelism every process runs the
    loaders. The last rename wins, with the same bytes as the others.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256 is not None and _sha256(target) != sha256:
            raise ValueError(
                f"{target} does not match the expected sha256; delete it to "
                "download it again"
            )
        return target

    # A name of this call's own rather than `tempfile.mkstemp`, whose file is
    # private to its owner and would stay so in a cache other users share.
    partial = target.with_name(f"{target.name}.{uuid.uuid4().hex}.part")
    try:
        urllib.request.urlretrieve(url, partial)
        if sha256 is not None and _sha256(partial) != sha256:
            raise ValueError(f"{url} does not match the expected sha256")
        partial.replace(target)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
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

    A corpus archive needs nothing else, so refusing the rest costs nothing
    and leaves no link semantics to reason about.
    """
    for member in members:
        if not (member.isfile() or member.isdir()):
            raise ValueError(
                f"refusing to extract archive member {member.name}: "
                "a corpus archive holds only files and directories"
            )
    return members


def extract(archive: Path, directory: Path) -> Path:
    """Unpack ``archive`` into ``directory`` once; return ``directory``.

    The archive is unpacked into a staging directory of this call's own and
    renamed onto ``directory`` once complete. A call that fails removes its
    staging directory. Several processes may extract the same archive at
    once: the first rename wins, and the others discard their copy.
    """
    if directory.exists():
        return directory

    # Named rather than `tempfile.mkdtemp`, for the permissions reason
    # `download` gives.
    staging = directory.with_name(f"{directory.name}.{uuid.uuid4().hex}.partial")
    staging.mkdir(parents=True)
    try:
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
        try:
            staging.rename(directory)
        except OSError:
            # Another process renamed its copy first.
            if not directory.is_dir():
                raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return directory
