"""Local entry point for checks run by GitHub Actions."""

from pathlib import Path

import nox

PYTHON_VERSIONS = ["3.10", "3.11", "3.12", "3.13"]
LINT_VERSION = "3.12"

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
    session.run(
        "pytest",
        "--cov=pyhighlights",
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
