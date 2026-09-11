# Zenodo upload — paste-ready

Four artifacts, **three published now** and one held. Everything below is meant
to be pasted into <https://zenodo.org/uploads/new>. Files are in `dist/`.

Reserve a DOI on each record before publishing (Zenodo's *Reserve DOI* button):
the reserved DOI can go into the pyhighlights docs in the same pass, and a
Zenodo record is versioned, so `v2` of an artifact keeps a concept DOI that
points at the newest.

Shared fields for all four:

- **Resource type:** Dataset
- **Creators:** Federico Ruggeri (University of Bologna,
  ORCID: *fill in*)
- **Version:** `v1`
- **Language:** English
- **Related identifiers:**
  - `https://github.com/nlp-unibo/pyhighlights` — *is supplement to* —
    Software
  - `https://pypi.org/project/pyhighlights/0.6.0/` — *is supplement to* —
    Software

---

## 1. `pyhighlights-r2a-beer-splits-v1.zip` — publish now

**Title**

> Zero-leakage split manifests for the R2A Beer aspects (pyhighlights)

**Licence:** CC0-1.0. The artifact is index data, counts and checksums
authored here, not the corpus — CC0 keeps a reproduction from stacking a second
attribution requirement on top of Bao et al.'s. Use CC-BY-4.0 instead if you
would rather be cited for the splits themselves.

**Keywords:** select-then-predict, rationalization, explainability, data
leakage, reproducibility, BeerAdvocate, R2A, pyhighlights

**Related identifiers, in addition to the shared ones**

- `https://people.csail.mit.edu/yujia/files/r2a/data.zip` — *references* —
  Dataset

**Description**

> Split manifests that pin the zero-leakage train/validation/test partition
> pyhighlights uses for the three Beer aspects of the R2A release of Bao et
> al., 2018 (*Deriving Machine Attention from Human Rationales*, EMNLP 2018).
>
> The distributed R2A splits overlap: about two thirds of each Beer validation
> split also appears in that aspect's training file. pyhighlights repairs this
> by walking the splits in the order test, validation, train, so the annotated
> split — the only one carrying per-token rationales, and the one highlight
> scores are reported on — is kept whole while training and validation give
> rows up.
>
> For each of `beer0`, `beer1` and `beer2` the manifest records the upstream
> row indices each split retains, the distributed and retained counts, label
> counts, the source URL and its SHA-256, and the repair policy that produced
> them. `beer0` retains 27,973 of 32,276 training rows, 6,388 of 6,392
> validation rows, and all 200 annotated test rows.
>
> **No review text is redistributed.** Fetch `data.zip` from the source below,
> verify it against the recorded digest, and index it with these manifests.
>
> Built with pyhighlights 0.6.0. The build is deterministic and the manifests
> round-trip: applying the recorded indices to the corpus as distributed
> reproduces the repaired splits exactly.
>
> Citation for the underlying corpus: Bao, Y., Chang, S., Yu, M., & Barzilay,
> R. (2018). Deriving Machine Attention from Human Rationales. EMNLP 2018.

---

## 2. `pyhighlights-r2a-hotel-splits-v1.zip` — publish now

**Title**

> Zero-leakage split manifests for the R2A Hotel aspects (pyhighlights)

**Licence:** CC0-1.0, as above.

**Keywords:** select-then-predict, rationalization, explainability, data
leakage, reproducibility, TripAdvisor, R2A, pyhighlights

**Related identifiers, in addition to the shared ones**

- `https://people.csail.mit.edu/yujia/files/r2a/data.zip` — *references* —
  Dataset

**Description**

> Split manifests that pin the zero-leakage train/validation/test partition
> pyhighlights uses for the three Hotel aspects of the R2A release of Bao et
> al., 2018 (*Deriving Machine Attention from Human Rationales*, EMNLP 2018).
>
> The distributed R2A splits overlap: every one of the 200 annotated rows of
> each Hotel aspect also appears in that aspect's training file. pyhighlights
> repairs this by walking the splits in the order test, validation, train, so
> the annotated split — the only one carrying per-token rationales — is kept
> whole while training and validation give rows up.
>
> For each of `hotel_Location`, `hotel_Service` and `hotel_Cleanliness` the
> manifest records the upstream row indices each split retains, the
> distributed and retained counts, label counts, the source URL and its
> SHA-256, and the repair policy. `hotel_Cleanliness` retains 125,868 of
> 150,098 training rows, 18,390 of 18,764 validation rows, and all 200
> annotated test rows.
>
> **No review text is redistributed.** Fetch `data.zip` from the source below,
> verify it against the recorded digest, and index it with these manifests.
>
> Built with pyhighlights 0.6.0. The build is deterministic and the manifests
> round-trip.
>
> Citation for the underlying corpus: Bao, Y., Chang, S., Yu, M., & Barzilay,
> R. (2018). Deriving Machine Attention from Human Rationales. EMNLP 2018.

---

## 3. `pyhighlights-eraser-movies-splits-v1.zip` — publish now

**Title**

> Zero-leakage split manifest for ERASER movies (pyhighlights)

**Licence:** CC0-1.0, as above.

**Keywords:** select-then-predict, rationalization, explainability, evidence
spans, data leakage, reproducibility, ERASER, pyhighlights

**Related identifiers, in addition to the shared ones**

- `https://www.eraserbenchmark.com/zipped/movies.tar.gz` — *references* —
  Dataset
- the ERASER ACL 2020 DOI — *is referenced by* — Publication. Candidate:
  `10.18653/v1/2020.acl-main.408`; **check it resolves before pasting** — it
  is from memory, not verified here.

**Description**

> The split manifest for the ERASER `movies` task as pyhighlights reads it —
> the only single-document ERASER task, and the one the select-then-predict
> literature reports highlight scores on.
>
> The manifest records the upstream row indices each split retains after the
> zero-leakage repair, the distributed and retained counts, label counts, the
> source URL and its SHA-256, and the repair policy. `movies` is almost clean
> as distributed: one training row is dropped as a duplicate, leaving 1,599
> train, 200 validation and 199 test rows, all three annotated with evidence
> spans.
>
> This artifact also pins the upstream archive's SHA-256,
> `66e18d4e6c9df9e9f5544572b0bfe92a39673f74ecbfc3859b46cedb2f5b2dee`, which
> the benchmark itself does not publish.
>
> **No document text is redistributed.** Fetch `movies.tar.gz` from the source
> below, verify it against the recorded digest, and index it with this
> manifest.
>
> Built with pyhighlights 0.6.0. The build is deterministic and the manifest
> round-trips.
>
> Citation for the underlying corpus: DeYoung, J., Jain, S., Rajani, N. F.,
> Lehman, E., Xiong, C., Socher, R., & Wallace, B. C. (2020). ERASER: A
> Benchmark to Evaluate Rationalized NLP Models. ACL 2020.

---

## 4. `pyhighlights-genspp-toy-v1.zip` + `toy_dataset.pkl` — **hold as draft**

Save this record as a draft and do **not** publish until both GenSPP authors
confirm the data licence. It is the one artifact that redistributes a dataset
rather than indexing one.

**Upload both files to this one record.** Zenodo serves each file of a record
at its own URL, and `GenSPPToyLoader` reads a pickle rather than an archive, so
the loader's default URL has to name `toy_dataset.pkl` directly; the zip is
there for a human who wants the README and manifest beside it.

**Title**

> The GenSPP synthetic toy corpus (pyhighlights reproduction artifact)

**Licence:** *pending the authors' answer.* CC-BY-4.0 is the natural choice for
a synthetic corpus with a paper behind it; CC0-1.0 if they would rather it
carry no conditions. Do not set this field before they answer.

**Keywords:** select-then-predict, rationalization, genetic algorithm,
synthetic corpus, reproducibility, GenSPP, pyhighlights

**Related identifiers, in addition to the shared ones**

- `https://github.com/nlp-unibo/gen-spp` — *is derived from* — Software
- the GenSPP ACL 2025 DOI — *is supplement to* — Publication *(fill in)*

**Description**

> The 10,000 synthetic sequences the GenSPP paper (ACL 2025) trains and
> reports on, as released in `nlp-unibo/gen-spp` at
> `genetic/data/toy_dataset.pkl`, republished so a reproduction can fetch them
> rather than be handed them.
>
> Each row is a twenty-character string over the lowercase alphabet with a
> three-character pattern hidden in it; the class is which pattern and the
> highlight is exactly where it sits. Tokens are characters, so the vocabulary
> is the alphabet. Class counts are 3,295 / 3,301 / 3,404.
>
> Splits are **not** stored. The loader derives them from the released
> baselines' scheme — the first 80% train, the rest test, a fifth of train
> sampled off for validation at seed 15000 — so the artifact is the corpus and
> the split is code.
>
> `pyhighlights_benchmarks.genspp2025.corpora.GenSPPToyLoader` reads
> `toy_dataset.pkl` directly. SHA-256:
> `2ee223d8aecd6ee9a585aa695a6fd5fd6f4c10c39fe6046ffb833d7ab44fb064`.
>
> Citation: the GenSPP paper, ACL 2025. Reference implementation:
> <https://github.com/nlp-unibo/gen-spp>.

---

## After the DOIs exist

Two one-line library changes, and no new machinery:

1. `R2ALoader.__init__` — `sha256: str | None = R2A_SHA256` with
   `R2A_SHA256 = "23fcb4cac883ec1de86d83a7747294d7fdae10061d3803fd4c34c930e66f25de"`.
2. `ERASERLoader.__init__` — `sha256: str | None = MOVIES_SHA256` with
   `66e18d4e6c9df9e9f5544572b0bfe92a39673f74ecbfc3859b46cedb2f5b2dee`. Note
   this is per-task, so it belongs beside `TASKS` rather than as one constant
   if a second task is ever supported.
3. Once the toy record is published: `GenSPPToyLoader.URL` becomes
   `https://zenodo.org/records/<id>/files/toy_dataset.pkl?download=1` and
   `sha256` defaults to
   `2ee223d8aecd6ee9a585aa695a6fd5fd6f4c10c39fe6046ffb833d7ab44fb064`. The
   `url is None` refusal in `download()` stays — a user pointing at a local
   pickle is still supported.

**No loader reads a manifest, and none needs to.** Pinning the upstream digest
plus a deterministic `remove_leakage` already yields the same splits on any
machine; the manifests are the published receipt that says which rows those
are, and what a reviewer checks a run against. A manifest-fetching code path
would be machinery for a guarantee the pin already gives.

Then update `docsrc/source/datasets.rst` and `docsrc/source/benchmarks.rst`
with the DOIs — `benchmarks.rst:89` currently says the toy loader has no
download URL yet.
