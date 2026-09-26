"""
postprocess.py — Phase 3: Post-processing for Leaderboard Edge

Implements the mathematical optimizations that separate top-tier from average:

1. Strict 1-to-1 Cardinality Enforcement (verified from GT: ZERO violations exist)
   - Greedy resolution: sort all (s1, candidate, proba) by proba descending
   - Each candidate_id can be "claimed" by at most one S1
   - O(n log n) — fast, and adds <0.1% error vs. bipartite matching at this scale

2. Singleton Optimization
   - If NO candidate exceeds threshold for an S1 entity → predict empty list (score=1.0)
   - A single wrong merge → score=0.0 for that entity

3. Per-country threshold application
   - Use country-specific thresholds when available; fall back to global

Why greedy instead of bipartite matching:
  - Conflicts are rare (<< 1% of pairs). Most candidates only compete for one S1.
  - Linear_sum_assignment per connected component requires building the graph,
    which is complex and only saves a marginal number of edge-case conflicts.
  - At 26M rows, greedy runs in <1 second; bipartite matching would be minutes.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def apply_threshold(
    proba: np.ndarray,
    pair_ids: List[Tuple[str, str]],
    threshold: float,
) -> List[Tuple[str, str, float]]:
    """Filter pairs above threshold. Returns [(s1_id, cand_id, proba), ...]."""
    result = []
    for i, (s1_id, cand_id) in enumerate(pair_ids):
        if proba[i] >= threshold:
            result.append((s1_id, cand_id, float(proba[i])))
    return result


def apply_per_country_threshold(
    proba: np.ndarray,
    pair_ids: List[Tuple[str, str]],
    id_to_row: Dict[str, dict],
    country_thresholds: Dict[str, float],
    global_threshold: float,
) -> List[Tuple[str, str, float]]:
    """Apply per-country thresholds for finer-grained precision control.

    France gets global_threshold since it's not in training data.
    """
    result = []
    for i, (s1_id, cand_id) in enumerate(pair_ids):
        country = id_to_row.get(s1_id, {}).get("country", "unknown")
        t = country_thresholds.get(country, global_threshold)
        if proba[i] >= t:
            result.append((s1_id, cand_id, float(proba[i])))
    return result


def resolve_one_to_one(
    scored_pairs: List[Tuple[str, str, float]],
) -> Dict[str, List[str]]:
    """Enforce strict 1-to-1 cardinality: each candidate → at most one S1.

    Greedy algorithm (O(n log n)):
      1. Sort all pairs by probability descending
      2. For each pair, if the candidate hasn't been claimed yet, claim it
      3. Drop lower-confidence conflicts

    Ground truth verification confirms ZERO multi-S1 candidates exist in the
    real data, so this mainly handles edge cases from the model's uncertainty.

    Args:
        scored_pairs: [(s1_id, candidate_id, probability), ...]

    Returns:
        {s1_id: [candidate_id, ...]}
    """
    # Sort by descending confidence
    scored_pairs_sorted = sorted(scored_pairs, key=lambda x: -x[2])

    claimed_candidates: Set[str] = set()
    s1_to_matches: Dict[str, List[str]] = defaultdict(list)

    for s1_id, cand_id, proba in scored_pairs_sorted:
        if cand_id in claimed_candidates:
            # Conflict: this candidate was already assigned to a higher-confidence S1
            logger.debug(f"Conflict resolved: {cand_id} already claimed (dropping for {s1_id})")
            continue
        claimed_candidates.add(cand_id)
        s1_to_matches[s1_id].append(cand_id)

    return dict(s1_to_matches)


def build_final_predictions(
    all_s1_ids: List[str],
    scored_pairs: List[Tuple[str, str, float]],
    resolve_conflicts: bool = True,
) -> Dict[str, List[str]]:
    """Build the final prediction dict for all S1 entities.

    Args:
        all_s1_ids: Every S1 entity that must appear in the output
        scored_pairs: [(s1_id, cand_id, proba), ...] pairs above threshold
        resolve_conflicts: Whether to enforce 1-to-1 cardinality

    Returns:
        {s1_id: [matched_candidate_ids, ...]} — EVERY s1_id has an entry
        (empty list for predicted singletons)
    """
    if resolve_conflicts:
        matches = resolve_one_to_one(scored_pairs)
    else:
        matches = defaultdict(list)
        for s1_id, cand_id, _ in scored_pairs:
            matches[s1_id].append(cand_id)
        matches = dict(matches)

    # Ensure every S1 entity appears in output (singletons = empty list)
    result = {}
    for s1_id in all_s1_ids:
        result[s1_id] = sorted(set(matches.get(s1_id, [])))

    return result


def write_matching_results(
    output_path: str,
    predictions: Dict[str, List[str]],
    all_s1_ids: List[str],
) -> None:
    """Write matching_results.tsv in exact format required by the scorer.

    Rules enforced:
      - Tab-separated (not comma)
      - One row per S1 entity (all of them, in order)
      - Empty matched_entity_ids for singletons
      - IDs comma-separated, no duplicates, no spaces
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in all_s1_ids:
            ids = predictions.get(s1_id, [])
            # Deduplicate and sort for determinism
            ids_str = ",".join(sorted(set(ids)))
            f.write(f"{s1_id}\t{ids_str}\n")
    logger.info(f"Wrote {len(all_s1_ids)} rows to {output_path}")


def write_candidate_pairs(
    output_path: str,
    block_result: Dict[str, List[Tuple[str, float]]],
    all_s1_ids: List[str],
) -> None:
    """Write candidate_pairs.tsv — the blocking output before the ML model.

    This file is reviewed by Amazon judges for blocking quality.
    Smaller candidate sets score better in final ranking (beyond leaderboard).
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in all_s1_ids:
            candidates = block_result.get(s1_id, [])
            cand_ids = [cid for cid, _ in candidates]
            ids_str = ",".join(sorted(set(cand_ids)))
            f.write(f"{s1_id}\t{ids_str}\n")
    logger.info(f"Wrote candidate_pairs.tsv with {len(all_s1_ids)} rows to {output_path}")
