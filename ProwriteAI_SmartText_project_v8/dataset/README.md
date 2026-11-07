# Dataset generation (synthetic, original)

* `v3/` — **dataset v3** (36,000 pairs, used by the notebook in `../data/`): `python v3/build3.py out_dir`. Extends v2 with 65 more frames (modal questions, daily-life verbs, typical learner errors, polite greetings); frames marked `*` in `frames.txt` are always kept in training.
* `v2/` — dataset v2 (30,000 pairs, `../data_v2/`): `python v2/build2.py out_dir`
  * `frames.txt` — 124 hand-written sentence frames (formal target template || informal templates)
  * `gen2.py` — slot lexicons, house-style greetings/closings, noise layer (contractions, chat abbreviations, confusable words, misspellings, typos, missing articles/punctuation)
  * `build2.py` — frame-level split before generation, validation (entities, numbers, negation, spelling, format), duplicate and near-duplicate removal
  * v2 also reuses the v1 scenario families (`families/`) where the informal variant stays close to the target.
* v1 (`build.py`, `families/`) — the first 12,000-pair dataset (`../data_v1/`).
* `verify.py out_dir` — independent CSV checks (columns, duplicates, cross-split leakage).

Generation is deterministic (fixed seeds): re-running `build2.py` reproduces the delivered CSVs exactly.
