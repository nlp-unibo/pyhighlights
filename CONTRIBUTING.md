# Contributing

## Setup

```bash
git clone git@github.com:nlp-unibo/pyhighlights.git
cd pyhighlights
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run local CI with `nox`:

| Command | Check |
|---|---|
| `nox` | Ruff and tests |
| `nox -s tests` | Tests with 75% branch-coverage gate |
| `nox -s transformers` | Optional Transformers installation |
| `nox -s docs` | Sphinx build with warnings as errors |
| `nox -s package` | Wheel/sdist build and metadata check |

## Branch and pull-request flow

1. Branch from updated `main` using `feat/`, `fix/`, `docs/`, or `chore/`.
2. Keep one logical change per branch; add tests and public documentation together.
3. Run `nox`, plus relevant optional session.
4. Open pull request and complete template.
5. Merge only when `All checks passed` succeeds and review is complete.
6. Prefer squash or rebase merge; delete merged branch.

Recommended `main` ruleset: require pull requests, one approval, resolved
conversations, linear history, and status check `All checks passed`; block force
pushes and branch deletion.

## Documentation

Edit `docsrc/source/`. `nox -s docs` writes generated HTML to
`docsrc/build/html`; generated files stay uncommitted. Merges to `main` deploy
that artifact through GitHub Pages.

## Release

1. Update `pyhighlights.__version__` in a pull request and merge it.
2. Tag matching commit: `git tag -a vX.Y.Z -m "pyhighlights X.Y.Z"`.
3. Push tag: `git push origin vX.Y.Z`.
4. `publish.yml` verifies tag/version, builds distributions, and publishes via
   PyPI Trusted Publishing.

Repository owner must enable GitHub Pages with **GitHub Actions** as source and
configure PyPI Trusted Publishing for owner `federicoruggeri`, repository
`pyhighlights`, workflow `publish.yml`, environment `pypi`.
