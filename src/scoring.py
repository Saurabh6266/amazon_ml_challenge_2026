"""
scoring.py — Official F_0.5 macro-average implementation.

Matches the exact formula in the problem statement:
    F_0.5 = (1.25 * P * R) / (0.25 * P + R)

Macro-averaged: compute F_0.5 per S1 entity, then average across ALL S1 entities
(including singletons — they score 1.0 if correctly predicted empty, 0.0 otherwise).
"""

from typing import Dict, Set


def f_beta_row(true_ids: Set[str], pred_ids: Set[str], beta: float = 0.5) -> float:
    """Compute F_beta for a single S1 entity."""
    # Both empty = correct singleton prediction = perfect score
    if not true_ids and not pred_ids:
        return 1.0
    # One is empty but not the other
    if not true_ids or not pred_ids:
        return 0.0
    tp = len(true_ids & pred_ids)
    if tp == 0:
        return 0.0
    precision = tp / len(pred_ids)
    recall = tp / len(true_ids)
    b2 = beta ** 2
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def macro_f_beta(
    gt_map: Dict[str, Set[str]],
    pred_map: Dict[str, Set[str]],
    beta: float = 0.5,
) -> float:
    """Macro-average F_beta over all S1 entities in gt_map.

    Args:
        gt_map: {s1_id: set of true matched IDs} — must include ALL S1 entities
                (singletons have empty sets, not missing keys)
        pred_map: {s1_id: set of predicted matched IDs}
        beta: 0.5 for precision-heavy weighting (competition default)

    Returns:
        Macro-averaged F_beta score in [0, 1]
    """
    scores = []
    for s1_id, true_ids in gt_map.items():
        pred_ids = pred_map.get(s1_id, set())
        scores.append(f_beta_row(true_ids, pred_ids, beta))
    return sum(scores) / len(scores) if scores else 0.0


def recall_ceiling(
    gt_map: Dict[str, Set[str]],
    block_result: Dict[str, list],
) -> float:
    """Fraction of true positives captured by the blocking stage.

    This is the absolute upper bound on recall your classifier can achieve.
    If this is < 0.95, fix blocking before touching the model.

    Args:
        gt_map: {s1_id: set of true matched IDs}
        block_result: {s1_id: list of (candidate_id, sim_score)}
    """
    total_positives = 0
    captured = 0
    for s1_id, true_ids in gt_map.items():
        if not true_ids:
            continue
        candidate_ids = {cid for cid, _ in block_result.get(s1_id, [])}
        total_positives += len(true_ids)
        captured += len(true_ids & candidate_ids)
    return captured / total_positives if total_positives > 0 else 0.0
