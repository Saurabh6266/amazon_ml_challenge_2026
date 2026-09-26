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
├── normalize.py     Language-agnostic NFKD+ASCII text normalization (no regex rules)
├── blocking.py      Phase 1: Country partition + TF-IDF char-3gram + FAISS ANN
├── features.py      Phase 2: RapidFuzz string distances + multilingual embeddings
├── scoring.py       Exact F_0.5 macro-average implementation (verified vs README)
├── train.py         LightGBM + GroupShuffleSplit + F_0.5 threshold tuning
├── postprocess.py   1-to-1 conflict resolution + TSV writers
└── pipeline.py      Main CLI orchestrator
```

## Key Design Decisions

1. **No country-specific regex** — all text features are pure math (cosine, edit distance)
2. **Char 3-grams** absorb abbreviation/suffix noise without hardcoded rules
3. **Apache 2.0 multilingual model** handles French test set natively
4. **Threshold tuned on macro F_0.5** (not AUC, not accuracy) on held-out val split
5. **Greedy 1-to-1 conflict resolution** enforces the dataset's cardinality constraint

## Output Files

Both go in `output/`:
- `matching_results.tsv` — final predictions, tab-separated
- `candidate_pairs.tsv` — blocking output, used for judge review of pipeline quality
