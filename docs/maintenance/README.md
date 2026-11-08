# Maintain a readable and auditable repository

[Repository map](../REPOSITORY_MAP.md) · [Agent entry point](../../AGENTS.md) · [History](../HISTORY.md)

Keep one current home for each case, one accepted-case catalogue, and one method guide per family. Introductory pages should route readers to the evidence, not repeat changing headline numbers or chronological progress notes.

## Retain what is still used

Preserve accepted inputs, original sources, row decisions and their hashes. Preserve current reports, posterior samples, candidate tables, full development attempts and unfavorable outcomes. A failed or superseded scientific alternative can still be necessary evidence; it is not automatically unused clutter.

When replacing a result, record the old commit and the new input/method identity. Retain an explicit local reference only when current calculation, audit or testing needs its files. Otherwise link to an immutable Git snapshot instead of copying an entire old report tree or adding redirect directories.

Historical provenance records retain the paths and hashes from their execution. A moved dependency should have a documented path resolver; do not edit its scientific contents just to make the filenames current. Generated reports with artifact manifests require deliberate checksum handling if their contents change.

## Verify an organization change

- Check every current data file and scientific artifact against its pre-change bytes, accounting for any explicitly moved reference.
- Confirm active solver code is unchanged, or validate numerical changes separately.
- Run tests and the local Markdown navigation checker.
- Run the relevant artifact verifier and inspect the staged Git tree, including files in nested `results/` directories. The ignore rule for scratch runs must be anchored to `/results/`.
- Leave unrelated working copies and untracked files untouched.

[2026-11-08 cleanup verification](cleanup-2026-11-08.json) records the retained content, removed collections and checks. The full pre-cleanup files remain in [Git history](../HISTORY.md).
