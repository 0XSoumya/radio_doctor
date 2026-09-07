# Project Status

## Current phase
Phase 1: Dataset Inspection, RES Metric Implementation & Baselines

## Best local RES
TBD (awaiting baseline evaluation)

## Best configuration
TBD

## Completed
- [x] Environment inspection and package installation (groq, pandas, numpy, scikit-learn, rapidfuzz, nbformat)
- [x] Unpacked dataset and validated CSV files
- [x] Repository structure, .gitignore, and requirements.txt initialized

## Current bottleneck
Establishing exact RES metric and running initial baselines on train.csv

## Next action
Implement `src/scoring.py` with exact RES normalization and weighted Levenshtein metric; verify with unit tests.

## Important artifacts
- data/train.csv (636 rows)
- data/test.csv (132 rows)
- data/sample_submission.csv (132 rows)

## Notes for continuation
- Model: `openai/gpt-oss-120b` via Groq
- Target: Local RES < 0.20 (Leaderboard reference: 0.23288)
