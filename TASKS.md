# matrainer — backlog

Open work for this project. Cross-cutting / multi-project initiatives live in the
workspace root `TASKS.md`. Completed items are not archived here — git history is the record.

- [ ] **Publish `0.1.0a1` — the steps after the release prep** @high — the package, metadata, CHANGELOG and `.github/workflows/release.yml` are ready (2026-09-27); what remains is outside the repo or needs a green run first: (1) make `Gearlux/matrainer` public — it is PRIVATE, so the PyPI page's Homepage/Issues/Changelog links and the README's absolute `docs/architecture.md` link all 404 for anyone else; (2) add the PyPI pending publisher (project `matrainer`, owner `Gearlux`, repo `matrainer`, workflow `release.yml`, environment `pypi`) and create the `pypi` environment on the GitHub repo; (3) `/ship` matrainer — `main` was last pushed 2026-05-28 and `ci.yml` runs only on `main`, so GitHub CI has not run on any of this code; (4) with CI green, `git tag v0.1.0a1 && git push origin v0.1.0a1`. @2026-09-27
