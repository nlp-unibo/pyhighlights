import hashlib
import io
import multiprocessing
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

    with pytest.raises(ValueError, match="delete it to download it again"):
        download(source.as_uri(), target, sha256="0" * 64)


def test_a_download_that_fails_its_checksum_is_not_cached(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("tampered")
    target = tmp_path / "cache" / "copy.txt"

    with pytest.raises(ValueError, match="does not match the expected sha256"):
        download(source.as_uri(), target, sha256="0" * 64)

    assert list(target.parent.iterdir()) == []


def race(work: tuple) -> str:
    kind, source, target = work
    try:
        if kind == "extract":
            extract(source, target)
        else:
            download(source.as_uri(), target)
        return "ok"
    except Exception as error:
        return f"{type(error).__name__}: {error}"


def test_several_processes_may_fetch_and_extract_one_target(tmp_path):
    """Under data parallelism every process runs the loaders at once."""
    archive = tmp_path / "corpus.tar.gz"
    with tarfile.open(archive, "w:gz") as target:
        for index in range(200):
            payload = b"x" * 20_000
            member = tarfile.TarInfo(f"corpus/rows{index}.txt")
            member.size = len(payload)
            target.addfile(member, io.BytesIO(payload))
    source = tmp_path / "vectors.bin"
    source.write_bytes(b"y" * 20_000_000)

    context = multiprocessing.get_context("spawn")
    with context.Pool(4) as pool:
        extracted = pool.map(race, [("extract", archive, tmp_path / "out")] * 4)
        fetched = pool.map(race, [("download", source, tmp_path / "cache" / "v")] * 4)

    assert extracted == ["ok"] * 4 and fetched == ["ok"] * 4
    assert len(list((tmp_path / "out" / "corpus").iterdir())) == 200
    assert (tmp_path / "cache" / "v").read_bytes() == source.read_bytes()
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "cache",
        "corpus.tar.gz",
        "out",
        "vectors.bin",
    ]
    assert [path.name for path in (tmp_path / "cache").iterdir()] == ["v"]


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


def test_a_tar_symlink_cannot_carry_a_write_outside_the_target(tmp_path):
    """Neither member name holds ``..``; together they escape without the fix.

    ``link -> ../outside`` is a name the name check accepts, and so is
    ``link/pwned.txt``. ``extractall`` creates the first and then follows it,
    which put the payload in ``outside/`` on every Python this package
    supports: 3.14 refuses it through its own default filter, 3.10 to 3.13 do
    not.
    """
    archive = tmp_path / "escape.tar"
    with tarfile.open(archive, "w") as target:
        link = tarfile.TarInfo("link")
        link.type = tarfile.SYMTYPE
        link.linkname = "../outside"
        target.addfile(link)
        payload = b"owned"
        member = tarfile.TarInfo("link/pwned.txt")
        member.size = len(payload)
        target.addfile(member, io.BytesIO(payload))
    (tmp_path / "outside").mkdir()

    with pytest.raises(ValueError, match="only files and directories"):
        extract(archive, tmp_path / "out")

    assert not (tmp_path / "outside" / "pwned.txt").exists()
    # And the refused extraction leaves no staging directory behind.
    assert sorted(path.name for path in tmp_path.iterdir()) == ["escape.tar", "outside"]


def test_a_tar_of_files_and_directories_still_unpacks(tmp_path):
    """What a corpus archive actually holds is untouched by the refusal."""
    archive = tmp_path / "corpus.tar.gz"
    with tarfile.open(archive, "w:gz") as target:
        directory = tarfile.TarInfo("docs")
        directory.type = tarfile.DIRTYPE
        directory.mode = 0o755
        target.addfile(directory)
        payload = b"a row"
        member = tarfile.TarInfo("docs/rows.txt")
        member.size = len(payload)
        target.addfile(member, io.BytesIO(payload))

    unpacked = extract(archive, tmp_path / "out")
    assert (unpacked / "docs" / "rows.txt").read_text() == "a row"


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


def test_a_failed_download_leaves_neither_a_corpus_nor_a_part_file(
    tmp_path, monkeypatch
):
    """An interrupted fetch has to be a missing file, not a truncated one."""
    target = tmp_path / "corpus.json"

    def fail(url, filename):
        Path(filename).write_bytes(b"half a corpus")
        raise OSError("connection reset")

    monkeypatch.setattr("urllib.request.urlretrieve", fail)
    with pytest.raises(OSError, match="connection reset"):
        download("https://example.invalid/corpus.json", target)

    assert not target.exists()
    assert list(tmp_path.iterdir()) == []
