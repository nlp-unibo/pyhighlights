# tools

Maintenance scripts. Nothing here is packaged — `[tool.setuptools.packages.find]`
includes only `pyhighlights*` and `pyhighlights_benchmarks*`, so none of this
ships in the wheel.

## `build_datasets.py`

Builds the Zenodo dataset artifacts that let a reproduction read the same rows
every time: zero-leakage split manifests for the R2A Beer and Hotel aspects and
for ERASER `movies`, plus the GenSPP synthetic toy corpus.

The build directory — the one holding `sources/` and `dist/` — is wherever you
run the script from, or whatever `--work-dir` names. It is deliberately not the
checkout: `sources/` is 690 MB of pinned upstream downloads.

```bash
cd <build directory>
python -m pip install -e '<checkout>'          # or use the project environment
python <checkout>/tools/build_datasets.py      # all four artifacts
python <checkout>/tools/build_datasets.py --skip-r2a    # toy only, no 162 MB fetch
python <checkout>/tools/build_datasets.py --selfcheck
```

`--selfcheck` builds twice and asserts the bytes agree, then asserts the movies
manifest reconstructs its own splits.

### What the artifacts are

| artifact | contents | licence |
|---|---|---|
| `pyhighlights-r2a-beer-splits-v1.zip` | split manifests for `beer0/1/2` | CC-BY-4.0 — no review text |
| `pyhighlights-r2a-hotel-splits-v1.zip` | split manifests for the three Hotel aspects | CC-BY-4.0 — no review text |
| `pyhighlights-eraser-movies-splits-v1.zip` | split manifest for ERASER `movies` | CC-BY-4.0 — no document text |
| `pyhighlights-genspp-toy-v1.zip` + `toy_dataset.pkl` | the complete released toy corpus | CC-BY-4.0, set by both GenSPP authors |

A manifest holds the upstream row indices each split retains after the
zero-leakage repair, the source URL and SHA-256, distributed and retained
counts, label counts, the repair policy, the licence and the citation. It holds
no corpus text, so its terms cover the indices and the policy rather than the
corpus, which carries whatever its own release carries.

The manifests are a published receipt rather than a runtime input. No loader
reads one and none needs to: pinning the upstream digest and running a
deterministic `remove_leakage` already reproduces the splits, so the manifest
is what a reviewer checks a run against.

### Pinned sources

- R2A — `https://people.csail.mit.edu/yujia/files/r2a/data.zip`,
  `23fcb4cac883ec1de86d83a7747294d7fdae10061d3803fd4c34c930e66f25de`
- ERASER movies — `https://www.eraserbenchmark.com/zipped/movies.tar.gz`,
  `66e18d4e6c9df9e9f5544572b0bfe92a39673f74ecbfc3859b46cedb2f5b2dee`. The
  benchmark does not publish this digest and `ERASERLoader` still defaults
  `sha256` to `None`.
- GenSPP toy — `nlp-unibo/gen-spp`, `genetic/data/toy_dataset.pkl`,
  `2ee223d8aecd6ee9a585aa695a6fd5fd6f4c10c39fe6046ffb833d7ab44fb064`

## `ZENODO.md`

Paste-ready metadata for the four Zenodo records — title, licence, keywords,
related identifiers, description — and the loader defaults the DOIs unlock.
