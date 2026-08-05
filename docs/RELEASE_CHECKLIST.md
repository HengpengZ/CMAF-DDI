# Public release checklist

Use this checklist before making the repository public or publishing a GitHub
Release.

## Required before public release

- [ ] Confirm that MIT is the intended source-code license.
- [ ] Review DrugBank terms for every committed or released label, mapping,
      structure, and derived feature file.
- [ ] Decide whether the existing files under `Knowledge Graph/raw_data/DRKG/`
      may remain public. Deleting them in a new commit does not remove them from
      Git history; use a history-rewrite process if required by the license.
- [ ] Place the four approved release-candidate artifacts in `data/DRKG/` or in
      an external archive, then run `scripts/validate_data.py --paper-profile
      --hash`.
- [ ] Compare the validator output with `data/artifact_manifest.json` and
      resolve every mismatch.
- [ ] Run `scripts/smoke_test.py` in a clean environment.
- [ ] Run at least the documented one-batch integration command on the actual
      data.
- [ ] Run the complete five-fold paper configuration and archive all outputs.
- [ ] Confirm that no checkpoint, credential, absolute local path, or licensed
      source file is accidentally staged.

## Repository settings

- [ ] Enable GitHub Actions and confirm the CI badge passes on `main`.
- [ ] Add a concise repository description and topics such as
      `drug-drug-interaction`, `multimodal-learning`, `pytorch`, and
      `bioinformatics`.
- [ ] Create a tagged release and attach only artifacts cleared for
      redistribution.
- [ ] Add an archival DOI if using Zenodo or another preservation service.
- [ ] Enable Issues or provide another supported contact path.

## After IEEE publication

- [ ] Replace the accepted-manuscript citation with the final IEEE citation.
- [ ] Add the DOI and article URL to `README.md` and `CITATION.cff`.
- [ ] Update the repository release notes without changing the hashes of the
      archived paper artifacts.
