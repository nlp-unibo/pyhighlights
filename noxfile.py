"""Local entry point for checks run by GitHub Actions."""

import os
from pathlib import Path

import nox

PYTHON_VERSIONS = ["3.10", "3.11", "3.12", "3.13"]
LINT_VERSION = "3.12"

#: Test workers. The suite is 153 mostly independent tests of well under a
#: second each, so it spent most of its wall clock on per-test setup rather
#: than on arithmetic -- 23s serial under coverage, 10s across eight workers.
#: Capped rather than left at ``-n auto``: every worker pays the torch import
#: and coverage startup again, so past eight the fixed cost outgrows what
#: another core saves. On a four-core CI runner this is four, which is what
#: ``auto`` would have chosen anyway.
WORKERS = str(min(8, os.cpu_count() or 1))

nox.options.default_venv_backend = "uv|virtualenv"
nox.options.error_on_missing_interpreters = False
nox.options.sessions = ["lint", "tests"]


@nox.session(python=LINT_VERSION)
def lint(session: nox.Session) -> None:
    session.install("-e", ".[dev]")
    session.run("ruff", "check", ".")
    session.run("ruff", "format", "--check", ".")


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    session.install("-e", ".[dev]")
    # Each worker is its own interpreter, which is also what keeps cinnamon's
    # global registry from being shared between tests that build it.
    session.run(
        "pytest",
        "-n",
        WORKERS,
        "--cov=pyhighlights",
        "--cov=pyhighlights_benchmarks",
        "--cov-branch",
        "--cov-fail-under=75",
        "--cov-report=term-missing",
    )


@nox.session(python=LINT_VERSION)
def transformers(session: nox.Session) -> None:
    session.install("-e", ".[dev,transformers]")
    session.run("python", "-c", "import transformers; import pyhighlights")
    session.run("pytest", "tests/test_transformer_configurations.py")


@nox.session(python=LINT_VERSION)
def docs(session: nox.Session) -> None:
    session.install("-e", ".[docs]")
    session.chdir("docsrc")
    session.run("bash", "build_docs.sh", external=True)


@nox.session(python=LINT_VERSION)
def package(session: nox.Session) -> None:
    session.install("build", "twine")
    session.run(
        "python",
        "-c",
        "import shutil; shutil.rmtree('dist', ignore_errors=True)",
    )
    session.run("python", "-m", "build")
    artifacts = [path.resolve() for path in Path("dist").glob("*")]
    session.run("twine", "check", "--strict", *(str(path) for path in artifacts))
    wheel = next(path for path in artifacts if path.suffix == ".whl")
    session.install(str(wheel))
    session.chdir("/tmp")
    session.run("python", "-c", "import pyhighlights; print(pyhighlights.__version__)")
