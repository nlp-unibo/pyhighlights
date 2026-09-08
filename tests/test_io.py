import hashlib
import tarfile
import zipfile
from pathlib import Path

import pytest

from pyhighlights.utility.io import cache_directory, download, extract


def test_cache_directory_follows_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("PYHIGHLIGHTS_CACHE", str(tmp_path / "elsewhere"))
    assert cache_directory() == tmp_path / "elsewhere"

    monkeypatch.delenv("PYHIGHLIGHTS_CACHE")
    assert cache_directory() == Path.home() / ".cache" / "pyhighlights"


def test_download_verifies_a_checksum_when_given_one(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("the artefact")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()

    target = download(source.as_uri(), tmp_path / "cache" / "copy.txt", sha256=digest)
    assert target.read_text() == "the artefact"

    # A second call reuses what is there rather than fetching again.
    source.write_text("something else")
    assert (
        download(source.as_uri(), target, sha256=digest).read_text() == "the artefact"
    )

    with pytest.raises(ValueError, match="does not match the expected sha256"):
        download(source.as_uri(), target, sha256="0" * 64)


def test_unsafe_archive_members_are_refused(tmp_path):
    """A zip or tar naming a path outside the target is never unpacked."""
    path = tmp_path / "escape.zip"
    with zipfile.ZipFile(path, "w") as target:
        target.writestr("../escaped.txt", "nope")

    with pytest.raises(ValueError, match="unsafe archive path"):
        extract(path, tmp_path / "out")

    escaping = tmp_path / "payload.txt"
    escaping.write_text("nope")
    tar = tmp_path / "escape.tar.gz"
    with tarfile.open(tar, "w:gz") as target:
        target.add(escaping, arcname="../escaped.txt")

    with pytest.raises(ValueError, match="unsafe archive path"):
        extract(tar, tmp_path / "tar-out")

    assert not (tmp_path / "escaped.txt").exists()


def test_only_zip_and_tar_archives_are_understood(tmp_path):
    archive = tmp_path / "corpus.bin"
    archive.write_bytes(b"not an archive")

    with pytest.raises(ValueError, match="neither a zip nor a tar archive"):
        extract(archive, tmp_path / "out")


def test_extract_unpacks_once(tmp_path):
    archive = tmp_path / "corpus.zip"
    with zipfile.ZipFile(archive, "w") as target:
        target.writestr("data/rows.txt", "a row")

    directory = extract(archive, tmp_path / "out")
    assert (directory / "data" / "rows.txt").read_text() == "a row"

    # A second call returns the existing directory untouched.
    (directory / "data" / "rows.txt").write_text("edited")
    assert extract(archive, directory) == directory
    assert (directory / "data" / "rows.txt").read_text() == "edited"
