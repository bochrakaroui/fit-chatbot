# Data governance and review

## Source policy

- Treat `raw_batches/` as immutable source material.
- Store deliberate edits and manual additions in `dataset/curated/overrides.json`.
- Store intentionally retired raw instructions in `dataset/curated/exclusions.json`.
- Never edit `fitness_qa.json` directly; regenerate it with `dataset/build_dataset.py`.
- Record the build SHA-256 from `dataset/reports/build_report.json` with every training run.

## Review workflow

`dataset/validate_dataset.py` creates `dataset/reports/review_queue.json`. Every queued record requires a qualified human reviewer before it can enter training. Reviewers should check correctness, appropriate uncertainty, contraindications, population assumptions, and whether a source supports the claim.

After an answer is approved or corrected, place the full record in `overrides.json` and adjust the review rule or metadata only when the reason is documented. Do not bypass the queue with `--include-needs-review` for a production model.

## Current automated gates

- Required schema and non-empty prompts/answers
- Unicode and whitespace normalization
- Case-insensitive instruction deduplication
- Repetitive boilerplate removal
- Deterministic build hash
- High- and medium-risk topic detection
- Heuristics for several known questionable exercise and fat-loss claims
- Prompt-family split isolation

Automated screening is not subject-matter approval. The current review queue remains a deliberate release constraint.
