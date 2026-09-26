"""
pipeline.py — Main Orchestration: Data → Blocking → Features → Train → Predict → Output

Usage:
    # Full pipeline (train + predict on test set):
    python3 src/pipeline.py --mode full

    # Validate blocking recall before training:
    python3 src/pipeline.py --mode validate_blocking

Key levers to tune (sweep on your validation split):
  --top-k         Candidates per S1 entity (default=20, try 10–30)
  --sim-floor     Min cosine to include a candidate (default=0.15, try 0.10–0.25)
  --threshold     Override global threshold (default=auto-tune)
  --no-embeddings Skip multilingual embeddings (faster, slightly lower recall on French)
"""

import argparse
import gc
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

# Add src/ to path so sub-imports work
sys.path.insert(0, str(Path(__file__).parent))

from blocking import run_blocking
from features import FEATURE_COLS, build_feature_matrix
from normalize import normalize_text
from postprocess import (
    apply_per_country_threshold,
    apply_threshold,
    build_final_predictions,
    write_candidate_pairs,
    write_matching_results,
)
from scoring import macro_f_beta, recall_ceiling
from train import (
    build_training_dataframe,
    train_model,
    tune_threshold_global,
    tune_threshold_per_country,
    save_model,
    load_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
TRAIN_S1 = ROOT / "dataset/train/train_source1.tsv"
TRAIN_S2 = ROOT / "dataset/train/train_source2.tsv"
TRAIN_S3 = ROOT / "dataset/train/train_source3.tsv"
TRAIN_GT = ROOT / "dataset/train/train_ground_truth.tsv"
TEST_S1 = ROOT / "dataset/test/test_source1.tsv"
TEST_S2 = ROOT / "dataset/test/test_source2.tsv"
TEST_S3 = ROOT / "dataset/test/test_source3.tsv"
OUTPUT_DIR = ROOT / "output"
MODEL_DIR = ROOT / "models"


def load_ground_truth(gt_path: str) -> Dict[str, Set[str]]:
    """Load ground truth into {s1_id: set(matched_ids)}. Singletons → empty set."""
    gt_map = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        f.readline()  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            s1_id = parts[0]
            matched = parts[1] if len(parts) > 1 else ""
            if matched.strip():
                gt_map[s1_id] = set(m for m in matched.split(",") if m.strip())
            else:
                gt_map[s1_id] = set()
    return gt_map


def flatten_block_result(
    block_result: Dict[str, List[Tuple[str, float]]]
) -> List[Tuple[str, str, float]]:
    """Flatten blocking output to a flat list of (s1_id, cand_id, sim) triples."""
    pairs = []
    for s1_id, candidates in block_result.items():
        for cand_id, sim in candidates:
            pairs.append((s1_id, cand_id, sim))
    return pairs


def run_full_pipeline(args):
    """Complete pipeline: blocking → features → train → inference → output."""
    t_start = time.time()
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    # ─── PHASE 1: BLOCKING ────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 1: BLOCKING (Country partition + TF-IDF + FAISS)")
    logger.info("=" * 60)

    logger.info("Running blocking on TRAINING data...")
    train_block_result, train_id_to_row = run_blocking(
        str(TRAIN_S1), str(TRAIN_S2), str(TRAIN_S3),
        top_k=args.top_k,
        sim_floor=args.sim_floor,
    )
    logger.info(f"Train blocking done. {len(train_block_result)} S1 entities.")

    # ─── VALIDATE RECALL CEILING ─────────────────────────────────────────────
    logger.info("Loading ground truth...")
    gt_map = load_ground_truth(str(TRAIN_GT))

    rc = recall_ceiling(gt_map, train_block_result)
    logger.info(f"RECALL CEILING (train blocking): {rc:.4f} ({rc*100:.2f}%)")
    if rc < 0.90:
        logger.warning(
            f"Recall ceiling {rc:.3f} is LOW. Your classifier cannot exceed this. "
            f"Consider increasing --top-k or decreasing --sim-floor."
        )

    # ─── PHASE 2a: FEATURE ENGINEERING (TRAIN) ────────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 2a: FEATURE ENGINEERING (training pairs)")
    logger.info("=" * 60)

    train_pairs_flat = flatten_block_result(train_block_result)
    logger.info(f"Total training candidate pairs: {len(train_pairs_flat)}")

    X_train_all, train_pair_ids = build_feature_matrix(
        train_pairs_flat,
        train_id_to_row,
        use_embeddings=not args.no_embeddings,
    )
    logger.info(f"Feature matrix shape: {X_train_all.shape}")

    # ─── PHASE 2b: ASSEMBLE LABELED DATAFRAME ─────────────────────────────────
    logger.info("Building labeled DataFrame...")
    # Assemble columns directly. Building one Python dict per candidate pair
    # multiplies memory use several-fold on this multi-million-pair dataset.
    labels = np.fromiter(
        (int(cand_id in gt_map.get(s1_id, set())) for s1_id, cand_id in train_pair_ids),
        dtype=np.uint8,
        count=len(train_pair_ids),
    )
    df = pd.DataFrame(X_train_all, columns=FEATURE_COLS, copy=False)
    df["source1_entity_id"] = [s1_id for s1_id, _ in train_pair_ids]
    df["candidate_entity_id"] = [cand_id for _, cand_id in train_pair_ids]
    df["label"] = labels
    del X_train_all, train_pairs_flat, train_pair_ids, labels
    gc.collect()

    n_pos = (df["label"] == 1).sum()
    n_neg = (df["label"] == 0).sum()
    logger.info(f"Labeled pairs: {len(df)} total, {n_pos} positive ({100*n_pos/len(df):.1f}%), {n_neg} negative")

    # ─── PHASE 2c: TRAIN MODEL ────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 2c: TRAINING LightGBM")
    logger.info("=" * 60)

    model, train_df, val_df = train_model(
        df,
        val_size=0.15,
        n_estimators=args.n_estimators,
        learning_rate=args.lr,
    )

    # ─── PHASE 2d: THRESHOLD TUNING ───────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 2d: THRESHOLD TUNING (macro F_0.5, NOT accuracy)")
    logger.info("=" * 60)

    if args.threshold is not None:
        global_threshold = args.threshold
        logger.info(f"Using user-specified threshold: {global_threshold}")
        country_thresholds = {}
    else:
        global_threshold = tune_threshold_global(model, val_df)
        if not args.no_per_country_threshold:
            country_thresholds = tune_threshold_per_country(model, val_df, train_id_to_row)
        else:
            country_thresholds = {}

    threshold_info = {
        "global": global_threshold,
        "per_country": country_thresholds,
    }
    save_model(model, threshold_info, str(MODEL_DIR))

    # ─── PHASE 3: INFERENCE ON TEST SET ───────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 3: BLOCKING + INFERENCE ON TEST SET")
    logger.info("=" * 60)

    logger.info("Running blocking on TEST data...")
    test_block_result, test_id_to_row = run_blocking(
        str(TEST_S1), str(TEST_S2), str(TEST_S3),
        top_k=args.top_k,
        sim_floor=args.sim_floor,
    )

    # Read all test S1 IDs (must appear in output in original order)
    all_test_s1_ids = []
    with open(str(TEST_S1), "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            eid = line.split("\t", 1)[0]
            if eid:
                all_test_s1_ids.append(eid)
    logger.info(f"Test blocking done. {len(all_test_s1_ids)} S1 test entities.")

    test_pairs_flat = flatten_block_result(test_block_result)
    logger.info(f"Total test candidate pairs: {len(test_pairs_flat)}")

    # Write candidate_pairs.tsv NOW (before ML scoring, as required)
    write_candidate_pairs(
        str(OUTPUT_DIR / "candidate_pairs.tsv"),
        test_block_result,
        all_test_s1_ids,
    )

    if not test_pairs_flat:
        logger.warning("No test candidate pairs! Writing empty matching_results.tsv.")
        write_matching_results(str(OUTPUT_DIR / "matching_results.tsv"), {}, all_test_s1_ids)
        return

    logger.info("Computing test features...")
    X_test, test_pair_ids = build_feature_matrix(
        test_pairs_flat,
        test_id_to_row,
        use_embeddings=not args.no_embeddings,
    )
    del test_pairs_flat
    gc.collect()

    logger.info("Running LightGBM inference...")
    test_proba = model.predict_proba(X_test)[:, 1]
    del X_test
    gc.collect()

    # Apply thresholds
    if country_thresholds:
        logger.info("Applying per-country thresholds...")
        above_threshold = apply_per_country_threshold(
            test_proba, test_pair_ids, test_id_to_row,
            country_thresholds, global_threshold,
        )
    else:
        logger.info(f"Applying global threshold {global_threshold:.2f}...")
        above_threshold = apply_threshold(test_proba, test_pair_ids, global_threshold)

    logger.info(f"{len(above_threshold)} pairs survive threshold.")

    # ─── PHASE 3b: POST-PROCESSING ────────────────────────────────────────────
    logger.info("Resolving 1-to-1 conflicts (greedy by confidence)...")
    predictions = build_final_predictions(
        all_test_s1_ids,
        above_threshold,
        resolve_conflicts=True,
    )

    total_matched = sum(len(v) for v in predictions.values())
    singletons = sum(1 for v in predictions.values() if not v)
    logger.info(f"Predicted matches: {total_matched}, singletons: {singletons}")

    write_matching_results(
        str(OUTPUT_DIR / "matching_results.tsv"),
        predictions,
        all_test_s1_ids,
    )

    elapsed = time.time() - t_start
    logger.info(f"Pipeline complete in {elapsed/60:.1f} minutes.")
    logger.info(f"Output: {OUTPUT_DIR}/matching_results.tsv + candidate_pairs.tsv")
    logger.info("Run: python3 utils/validate_submission.py to verify before submitting!")


def run_validate_blocking(args):
    """Just check the recall ceiling of your blocking settings."""
    logger.info(f"Validating blocking: top_k={args.top_k}, sim_floor={args.sim_floor}")
    block_result, _ = run_blocking(
        str(TRAIN_S1), str(TRAIN_S2), str(TRAIN_S3),
        top_k=args.top_k,
        sim_floor=args.sim_floor,
    )
    gt_map = load_ground_truth(str(TRAIN_GT))
    rc = recall_ceiling(gt_map, block_result)
    total_pairs = sum(len(v) for v in block_result.values())
    logger.info(f"Recall ceiling: {rc:.4f}")
    logger.info(f"Total candidate pairs: {total_pairs}")
    logger.info(f"Avg candidates per S1: {total_pairs/max(len(block_result),1):.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Amazon ML Challenge 2026 — Entity Resolution Pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "validate_blocking"],
        default="full",
        help="Pipeline mode (default: full)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Max candidate pairs per S1 entity (default: 20). Smaller = smaller candidate_pairs.tsv.",
    )
    parser.add_argument(
        "--sim-floor",
        type=float,
        default=0.15,
        help="Min cosine similarity to include a candidate (default: 0.15)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override global classification threshold (default: auto-tuned on F_0.5)",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=500,
        help="LightGBM n_estimators (default: 500; early stopping will cap this)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.05,
        help="LightGBM learning rate (default: 0.05)",
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Skip multilingual embedding feature (faster, slightly weaker on French)",
    )
    parser.add_argument(
        "--no-per-country-threshold",
        action="store_true",
        help="Skip per-country threshold tuning (use single global threshold)",
    )

    args = parser.parse_args()

    if args.mode == "validate_blocking":
        run_validate_blocking(args)
    else:
        run_full_pipeline(args)


if __name__ == "__main__":
    main()
