# ML Challenge 2026: Business Entity Resolution — Complete Pipeline README

## Reproduction Instructions

### 1. Setup Environment
```bash
pip install -r requirements.txt
```

### 2. Run the Full Pipeline (Train → Predict → Output)
```bash
# Standard run (top_k=20, auto-tune threshold)
python3 src/pipeline.py --mode full

# Faster run (skip embeddings, lower recall on French)
python3 src/pipeline.py --mode full --no-embeddings

# Tune candidate size (sweep top_k first):
python3 src/pipeline.py --mode validate_blocking --top-k 10 --sim-floor 0.20
python3 src/pipeline.py --mode validate_blocking --top-k 15 --sim-floor 0.15
python3 src/pipeline.py --mode validate_blocking --top-k 20 --sim-floor 0.15
```

### 3. Validate Output Before Submitting (mandatory — 5 submissions/day limit!)
```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

## Pipeline Architecture

```
src/
├── normalize.py     Language-agnostic NFKD+ASCII text normalization
├── blocking.py      Phase 1: Country partition + word TF-IDF + sparse top-K cosine
├── features.py      Phase 2: RapidFuzz string distances + multilingual embeddings
├── scoring.py       Exact F_0.5 macro-average implementation (verified vs README)
├── train.py         LightGBM + GroupShuffleSplit + F_0.5 threshold tuning
├── postprocess.py   1-to-1 conflict resolution + TSV writers
└── pipeline.py      Main CLI orchestrator
```

## Key Design Decisions

1. Country partitioning avoids cross-country comparisons.
2. The current word-level blocker is sparse and avoids a dense all-pairs matrix,
   but can miss pairs that share no exact normalized word.
3. Embeddings are optional and computationally expensive at this scale; their
   benefit must be established with held-out validation.
4. The ML threshold is tuned on macro F_0.5 for candidate-bearing held-out S1s.
5. Training labels show no target ID reused across S1 entities. Greedy conflict
   resolution uses that observed pattern; it is not an explicit challenge rule.

The baseline threshold 0.70 is experimental, not a validated optimum. The
baseline script does not calculate a score; only the leaderboard can report it.

## Output Files

Both go in `output/`:
- `matching_results.tsv` — final predictions, tab-separated
- `candidate_pairs.tsv` — blocking output, used for judge review of pipeline quality
