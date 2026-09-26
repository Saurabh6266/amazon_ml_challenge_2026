"""
train.py — Phase 2: LightGBM Classifier Training + Threshold Tuning

Key design decisions:
  1. GroupShuffleSplit by source1_entity_id — never split a S1 entity's
     candidates across train/val to prevent data leakage.
  2. scale_pos_weight compensates for heavy class imbalance (~26% positives).
  3. Threshold is tuned on macro F_0.5 (NOT accuracy, NOT AUC-ROC) on a
     held-out validation split — this is the actual competition metric.
  4. Per-country thresholds can be tried for additional gains (France will
     likely differ from US/India since model never saw French positives).
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from features import FEATURE_COLS
from scoring import macro_f_beta

logger = logging.getLogger(__name__)


def build_training_dataframe(
    block_result: Dict[str, List[Tuple[str, float]]],
    id_to_row: Dict[str, dict],
    gt_map: Dict[str, Set[str]],
    feature_matrix_X: np.ndarray,
    pair_ids: List[Tuple[str, str]],
) -> pd.DataFrame:
    """Assemble the labeled training DataFrame.

    Args:
        block_result: Output of blocking stage
        id_to_row: Global entity_id → row dict
        gt_map: {s1_id: set of true match IDs} from ground truth
        feature_matrix_X: Pre-computed feature matrix (n_pairs × n_features)
        pair_ids: [(s1_id, cand_id)] in same row order as feature_matrix_X

    Returns:
        DataFrame with columns: FEATURE_COLS + [source1_entity_id, candidate_entity_id, label]
    """
    rows = []
    for i, (s1_id, cand_id) in enumerate(pair_ids):
        true_matches = gt_map.get(s1_id, set())
        label = 1 if cand_id in true_matches else 0
        row = {col: float(feature_matrix_X[i, j]) for j, col in enumerate(FEATURE_COLS)}
        row["source1_entity_id"] = s1_id
        row["candidate_entity_id"] = cand_id
        row["label"] = label
        rows.append(row)  # MUST be inside loop
    return pd.DataFrame(rows)


def train_model(
    df: pd.DataFrame,
    val_size: float = 0.15,
    random_state: int = 42,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
) -> Tuple[lgb.LGBMClassifier, pd.DataFrame, pd.DataFrame]:
    """Train LightGBM with group-aware train/val split.

    Args:
        df: Labeled training DataFrame from build_training_dataframe()
        val_size: Fraction of S1 entities to hold out for validation
        random_state: For reproducibility
        n_estimators: Number of boosting rounds
        learning_rate: LightGBM learning rate

    Returns:
        (model, train_df, val_df)
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    train_idx, val_idx = next(gss.split(df, groups=df["source1_entity_id"]))
    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()

    n_pos = (train_df["label"] == 1).sum()
    n_neg = (train_df["label"] == 0).sum()
    spw = n_neg / max(n_pos, 1)
    logger.info(f"Train: {len(train_df)} pairs, {n_pos} pos, {n_neg} neg, scale_pos_weight={spw:.1f}")
    logger.info(f"Val:   {len(val_df)} pairs, {(val_df['label']==1).sum()} pos")

    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=7,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        scale_pos_weight=spw,
        n_jobs=-1,
        random_state=random_state,
        verbose=-1,
    )
    model.fit(
        train_df[FEATURE_COLS],
        train_df["label"],
        eval_set=[(val_df[FEATURE_COLS], val_df["label"])],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(50)],
    )
    logger.info(f"Best iteration: {model.best_iteration_}")
    return model, train_df, val_df


def tune_threshold_global(
    model: lgb.LGBMClassifier,
    val_df: pd.DataFrame,
    thresholds: Optional[np.ndarray] = None,
) -> float:
    """Find the global probability threshold maximizing macro F_0.5.

    Because F_0.5 weighs precision 2x over recall, expect best threshold
    to land at 0.70–0.85 (not the naive 0.50).

    Args:
        model: Trained LightGBM model
        val_df: Validation DataFrame with label column and source1_entity_id
        thresholds: Array of thresholds to sweep; defaults to 0.30–0.95 step 0.01

    Returns:
        Best threshold (float)
    """
    if thresholds is None:
        thresholds = np.arange(0.30, 0.96, 0.01)

    val_df = val_df.copy()
    val_df["proba"] = model.predict_proba(val_df[FEATURE_COLS])[:, 1]

    # Build ground truth map from val set (singletons = those S1 with no positive rows)
    gt_map = (
        val_df[val_df["label"] == 1]
        .groupby("source1_entity_id")["candidate_entity_id"]
        .apply(set)
        .to_dict()
    )
    for s1_id in val_df["source1_entity_id"].unique():
        gt_map.setdefault(s1_id, set())

    best_t, best_score = 0.5, -1.0
    results = []
    for t in thresholds:
        pred_df = val_df[val_df["proba"] >= t]
        pred_map = (
            pred_df.groupby("source1_entity_id")["candidate_entity_id"].apply(set).to_dict()
        )
        score = macro_f_beta(gt_map, pred_map)
        results.append((t, score))
        if score > best_score:
            best_score, best_t = score, float(t)

    logger.info(f"Threshold sweep complete. Best: t={best_t:.2f} → F_0.5={best_score:.4f}")
    # Log top-5 thresholds for transparency
    results.sort(key=lambda x: -x[1])
    for t, s in results[:5]:
        logger.info(f"  t={t:.2f} → F_0.5={s:.4f}")

    return best_t


def tune_threshold_per_country(
    model: lgb.LGBMClassifier,
    val_df: pd.DataFrame,
    id_to_row: Dict[str, dict],
    thresholds: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Per-country threshold tuning for fine-grained optimization.

    France is absent from training, so we cannot tune it — it gets the global threshold.
    US and India each get their own optimal threshold.

    Returns:
        {country: optimal_threshold}
    """
    if thresholds is None:
        thresholds = np.arange(0.30, 0.96, 0.01)

    val_df = val_df.copy()
    val_df["proba"] = model.predict_proba(val_df[FEATURE_COLS])[:, 1]
    val_df["country"] = val_df["source1_entity_id"].map(
        lambda sid: id_to_row.get(sid, {}).get("country", "unknown")
    )

    country_thresholds = {}
    global_t = tune_threshold_global(model, val_df, thresholds)

    for country in val_df["country"].unique():
        cdf = val_df[val_df["country"] == country]
        if len(cdf) < 1000:
            country_thresholds[country] = global_t
            continue

        gt_map = (
            cdf[cdf["label"] == 1]
            .groupby("source1_entity_id")["candidate_entity_id"]
            .apply(set)
            .to_dict()
        )
        for s1_id in cdf["source1_entity_id"].unique():
            gt_map.setdefault(s1_id, set())

        best_t, best_score = global_t, -1.0
        for t in thresholds:
            pred_df = cdf[cdf["proba"] >= t]
            pred_map = (
                pred_df.groupby("source1_entity_id")["candidate_entity_id"].apply(set).to_dict()
            )
            score = macro_f_beta(gt_map, pred_map)
            if score > best_score:
                best_score, best_t = score, float(t)

        country_thresholds[country] = best_t
        logger.info(f"  [{country}] optimal threshold={best_t:.2f}, F_0.5={best_score:.4f}")

    return country_thresholds


def save_model(model, threshold_info, output_dir: str = "models"):
    """Save trained model and threshold config."""
    Path(output_dir).mkdir(exist_ok=True)
    model.booster_.save_model(f"{output_dir}/lgbm_model.txt")
    with open(f"{output_dir}/threshold_info.pkl", "wb") as f:
        pickle.dump(threshold_info, f)
    logger.info(f"Model saved to {output_dir}/")


def load_model(output_dir: str = "models"):
    """Load saved model and threshold config."""
    import lightgbm as lgb

    model = lgb.Booster(model_file=f"{output_dir}/lgbm_model.txt")
    with open(f"{output_dir}/threshold_info.pkl", "rb") as f:
        threshold_info = pickle.load(f)
    return model, threshold_info
