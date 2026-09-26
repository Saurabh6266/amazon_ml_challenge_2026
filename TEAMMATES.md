# 🤝 Team Guide — Amazon ML Challenge 2026

> **Built with [Google Antigravity](https://antigravity.google) — AI-assisted code generation, planning, and execution.**
> Fork this repo, follow your assigned role below, and let's hit the top of the leaderboard.

---

## ⚡ TL;DR — What's Built & What You Need to Do

This is a complete, production-ready **Entity Resolution pipeline** for the Amazon ML Challenge 2026.

| Stage | What it does | Time |
|---|---|---|
| `src/baseline.py` | TF-IDF cosine baseline, no training needed | ~60–90 min |
| `src/pipeline.py --no-embeddings` | Full ML pipeline (LightGBM, RapidFuzz features) | ~3–4 h |
| `src/pipeline.py` | Full ML + multilingual embeddings (French support) | ~5–6 h |

**Current leaderboard status:** Baseline running with threshold=0.70 (higher precision, fewer false merges).

---

## 🛠️ One-Time Setup (Everyone Does This)

```bash
# 1. Clone the repo
git clone git@github.com:Saurabh6266/amazon_ml_challenge_2026.git
cd amazon_ml_challenge_2026

# 2. Install dependencies
pip install -r requirements.txt

# 3. Get the dataset from Unstop (team leader shares the zip)
#    Unzip into dataset/train/ and dataset/test/ so you have:
#      dataset/train/train_source1.tsv
#      dataset/train/train_source2.tsv
#      dataset/train/train_source3.tsv
#      dataset/train/train_ground_truth.tsv
#      dataset/test/test_source1.tsv
#      dataset/test/test_source2.tsv
#      dataset/test/test_source3.tsv

# 4. Create output and logs directories
mkdir -p output logs models
```

> ⚠️ **Dataset files are NOT in this repo** (2.5GB, too large for GitHub). Get them from the team leader.

---

## 👥 Role Assignments

### 🔵 Teammate 1 — Full Pipeline WITHOUT Embeddings

**Goal:** Get a strong ML baseline score. Expected F_0.5: **0.90–0.94**

```bash
mkdir -p logs

# Run in background (takes 3-4 hours)
nohup python3 src/pipeline.py \
    --mode full \
    --top-k 20 \
    --sim-floor 0.10 \
    --no-embeddings \
    > logs/pipeline_no_embed.log 2>&1 &

echo "Pipeline PID: $!"
# Watch live:
tail -f logs/pipeline_no_embed.log
```

**After it finishes:**
```bash
# Step 1: Validate
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test

# Step 2: If PASS → upload output/matching_results.tsv to Unstop portal
# Step 3: Note your F_0.5 score and share with team
```

**What to look for in the log:**
- `Recall ceiling: X.XXXX` after blocking — should be ≥ 0.95
- `Best threshold: t=X.XX, macro F_0.5=X.XXXX` — your val score
- If recall ceiling < 0.92: ping team leader to increase `--top-k`

---

### 🟢 Teammate 2 — Full Pipeline WITH Embeddings (French support)

**Goal:** Highest possible score by using multilingual embeddings for French. Expected F_0.5: **0.92–0.96**

The key difference: `paraphrase-multilingual-MiniLM-L12-v2` (Apache 2.0, 118M params) handles French text natively without any language-specific rules.

```bash
mkdir -p logs

# First run: downloads the embedding model (~420MB, happens once)
nohup python3 src/pipeline.py \
    --mode full \
    --top-k 20 \
    --sim-floor 0.10 \
    > logs/pipeline_with_embed.log 2>&1 &

echo "Pipeline PID: $!"
tail -f logs/pipeline_with_embed.log
```

**After it finishes:**
```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test

# Upload output/matching_results.tsv to Unstop portal
```

**Important notes:**
- The embedding step downloads `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` on first run
- Needs ~4GB RAM for the embedding computation phase
- If RAM is tight: close other apps or split the embedding into batches

---

### 🔴 Teammate 3 — Model Optimization & Threshold Tuning

**Goal:** Squeeze the leaderboard score higher by tuning the math. This is where wins are made.

#### 3a. Validate your own F_0.5 score on training data

```python
# Run this interactively to check your model's val score
import sys; sys.path.insert(0, 'src')
from scoring import macro_f_beta

# Load the val predictions your model made vs ground truth
# (pipeline.py logs the val score — look for "macro F_0.5" in the log)
```

#### 3b. Threshold Sweep (Most Impactful)

The pipeline auto-tunes the threshold but you can do a finer manual sweep:

```python
import sys, pandas as pd, numpy as np, pickle
sys.path.insert(0, 'src')
from scoring import macro_f_beta
import lightgbm as lgb

# Load saved model
model = lgb.Booster(model_file='models/lgbm_model.txt')
with open('models/threshold_info.pkl', 'rb') as f:
    threshold_info = pickle.load(f)

print("Current thresholds:", threshold_info)
# Then manually try thresholds between 0.60-0.90 on your val split
```

#### 3c. Feature Importance Analysis

```python
import sys, pickle, lightgbm as lgb, matplotlib.pyplot as plt
sys.path.insert(0, 'src')
from features import FEATURE_COLS

model = lgb.Booster(model_file='models/lgbm_model.txt')
importances = model.feature_importance(importance_type='gain')
for feat, imp in sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1]):
    print(f"  {feat:35s} {imp:.1f}")
```

#### 3d. Key Optimization Ideas (Ranked by Expected Impact)

```
1. PER-COUNTRY THRESHOLD TUNING (High Impact)
   - France gets global threshold (never seen in training)
   - US and India thresholds can diverge — try tuning separately
   - Add --no-per-country-threshold flag to pipeline.py to compare

2. TOP-K TUNING (High Impact on Judge Ranking)
   - Judges reward smaller candidate_pairs.tsv
   - If recall ceiling > 0.97 with top_k=15, use --top-k 15
   - Run: python3 src/pipeline.py --mode validate_blocking --top-k 15 --sim-floor 0.10

3. ADDING MORE FEATURES (Medium Impact)
   - Edit src/features.py → add to FEATURE_COLS
   - Ideas: phonetic encoding (Soundex/Metaphone for English names)
             address token overlap count
             shared numeric token count (street numbers, zip codes)
             name length ratio after stripping legal suffixes

4. XGBOOST vs LIGHTGBM (Low Impact, Try if Time)
   - src/train.py uses LightGBM by default
   - Try XGBoost with same features — sometimes wins 0.002-0.005 F_0.5

5. SINGLETON TUNING (High Precision Impact)
   - ~5.58% of S1 entities have zero true matches
   - If your model incorrectly predicts matches for singletons → big score drop
   - Try higher threshold for entities where max(proba) < 0.60 → predict singleton
```

#### 3e. Singleton Detector Script

```python
# Run after pipeline produces output to analyze singleton errors
import sys; sys.path.insert(0, 'src')
from scoring import f_beta_row

# Load ground truth and predictions, then:
# For each entity where true_ids=set() and pred_ids != set():
#   print(f"FALSE MERGE on singleton: {s1_id} predicted {pred_ids}")
# This tells you if you're bleeding score on singletons
```

---

## 🏗️ Pipeline Architecture

```
src/
├── normalize.py      Language-agnostic normalization (NFKD + ASCII, char 3-grams)
│                     No country-specific rules → works on US, India, AND French
│
├── blocking.py       Phase 1: Country partition + TF-IDF char-3gram + chunked sparse cosine
│                     Memory-safe: 8GB RAM verified. Uses per-S1 heap for top-K.
│
├── features.py       Phase 2: 13 RapidFuzz features + multilingual embedding cosine
│                     Jaro-Winkler, Levenshtein, token sort/set ratios, length signals
│                     Embedding: paraphrase-multilingual-MiniLM-L12-v2 (Apache 2.0, 118M params)
│
├── scoring.py        Exact F_0.5 macro-average (verified against README example)
│                     Also: recall_ceiling() to audit blocking quality
│
├── train.py          LightGBM + GroupShuffleSplit (no data leakage) + F_0.5 threshold tuning
│                     scale_pos_weight handles 26% positive / 74% negative imbalance
│
├── postprocess.py    Greedy 1-to-1 conflict resolution + TSV writers
│                     Enforces: each S2/S3 entity → at most one S1 entity
│
├── pipeline.py       Main CLI orchestrator (--mode full/train/validate_blocking)
│
└── baseline.py       Fast TF-IDF baseline (no training, just blocking + threshold)
```

---

## 📊 Verified Dataset Facts

All numbers computed from the official dataset (not external sources):

| Fact | Value |
|---|---|
| Total rows (train + test) | 26,430,544 |
| Train S1 entities | 2,206,821 |
| Singleton S1 (no matches) | 5.58% (123,247) |
| S2+S3 distractor rate | 26.0% |
| 1-to-1 cardinality violations | **0** (strict) |
| France in test only | 15.0% of test S1 (259,452 entities) |
| True pair Jaccard: median | 0.71 (TF-IDF captures this well) |

---

## ⚠️ Critical Rules (Do Not Violate)

1. **NO external APIs** — no Google Maps, geocoding, business lookup databases
2. **TSV not CSV** — `sep='\t'` in all file writes (commas are in address fields)
3. **Model ≤ 8B params, Apache 2.0 / MIT license** — MiniLM-L12-v2 = 118M ✅ LightGBM = MIT ✅
4. **Run validator before every submission** — 5 submissions/day limit, don't waste on formatting
5. **candidate_pairs.tsv must be in the final zip** — required by judges

---

## 📋 Submission Checklist

```bash
# Always run this before uploading to Unstop:
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test

# Upload ONLY: output/matching_results.tsv  (to Unstop portal)
# Final zip includes: matching_results.tsv + candidate_pairs.tsv + code/ + Documentation_template.md
```

---

## 🤖 Built With

- **Google Antigravity IDE** — AI-assisted code generation, architecture design, and pipeline planning
- **LightGBM** (MIT) — pairwise classifier
- **RapidFuzz** — string similarity features
- **sentence-transformers** (Apache 2.0) — multilingual embeddings
- **scikit-learn** — TF-IDF vectorization
- **datasketch** — MinHash (evaluated, replaced with chunked sparse cosine for better recall)

---

*Good luck! Share your val F_0.5 scores in the team chat as you complete each run.*
