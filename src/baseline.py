"""
baseline.py — Fast Leaderboard Baseline (No ML Training Required)

Strategy:
  - Run blocking on TEST set only (no training data needed)
  - Use TF-IDF cosine similarity directly as the prediction signal
  - Any candidate with cosine >= threshold → predicted match
  - Apply 1-to-1 conflict resolution (greedy)
  - Write matching_results.tsv + candidate_pairs.tsv

Why this is useful:
  - Runs in ~60-90 min instead of 4-5 hours
  - Gets you a real leaderboard score to benchmark against
  - Reveals how well pure TF-IDF performs (usually 0.70-0.85 F_0.5)
  - The full ML pipeline will then improve on this by 0.10-0.15 points

Expected score: 0.70-0.85 F_0.5 (depends on threshold)
Full pipeline target: 0.90-0.96 F_0.5
"""

import argparse
import gc
import heapq
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent))
from blocking import run_blocking
from postprocess import build_final_predictions, write_candidate_pairs, write_matching_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
TEST_S1  = ROOT / "dataset/test/test_source1.tsv"
TEST_S2  = ROOT / "dataset/test/test_source2.tsv"
TEST_S3  = ROOT / "dataset/test/test_source3.tsv"
OUTPUT   = ROOT / "output"


def main():
    parser = argparse.ArgumentParser(description="Fast TF-IDF baseline for leaderboard submission")
    parser.add_argument("--top-k",    type=int,   default=10,   help="Candidates per S1 (default 10, lower = faster + smaller candidate_pairs.tsv)")
    parser.add_argument("--sim-floor",type=float, default=0.10, help="Min cosine to include candidate (default 0.10)")
    parser.add_argument("--threshold",type=float, default=0.50, help="Min cosine to PREDICT a match (default 0.50 — tune higher for precision)")
    args = parser.parse_args()

    OUTPUT.mkdir(exist_ok=True)
    t_start = time.time()

    # ── Step 1: Block on test set ─────────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("STEP 1: Blocking on TEST data")
    logger.info(f"  top_k={args.top_k}, sim_floor={args.sim_floor}")
    logger.info("=" * 55)

    block_result, _ = run_blocking(
        str(TEST_S1), str(TEST_S2), str(TEST_S3),
        top_k=args.top_k,
        sim_floor=args.sim_floor,
    )

    # Read all test S1 IDs in original order
    all_test_s1_ids = []
    with open(str(TEST_S1), "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            eid = line.split("\t", 1)[0]
            if eid:
                all_test_s1_ids.append(eid)

    logger.info(f"Blocking complete. {len(all_test_s1_ids):,} S1 entities.")
    total_cands = sum(len(v) for v in block_result.values())
    logger.info(f"Total candidate pairs: {total_cands:,}, avg {total_cands/max(len(all_test_s1_ids),1):.1f}/S1")

    # ── Step 2: Write candidate_pairs.tsv (blocking output, required in zip) ─
    logger.info("Writing candidate_pairs.tsv...")
    write_candidate_pairs(
        str(OUTPUT / "candidate_pairs.tsv"),
        block_result,
        all_test_s1_ids,
    )

    # ── Step 3: Apply threshold to cosine scores → predicted matches ──────────
    logger.info(f"Applying match threshold: cosine >= {args.threshold}")

    # Flat list of (s1_id, cand_id, cosine_sim) above threshold
    above_threshold: List[Tuple[str, str, float]] = []
    for s1_id, candidates in block_result.items():
        for cand_id, sim in candidates:
            if sim >= args.threshold:
                above_threshold.append((s1_id, cand_id, sim))

    logger.info(f"Pairs above threshold: {len(above_threshold):,}")

    # ── Step 4: Greedy 1-to-1 conflict resolution ─────────────────────────────
    # Sort by descending confidence, claim each candidate for the highest-sim S1
    logger.info("Resolving 1-to-1 conflicts (greedy by cosine score)...")
    above_threshold.sort(key=lambda x: -x[2])
    claimed: Set[str] = set()
    s1_to_matches: Dict[str, List[str]] = defaultdict(list)
    for s1_id, cand_id, _ in above_threshold:
        if cand_id not in claimed:
            claimed.add(cand_id)
            s1_to_matches[s1_id].append(cand_id)

    # Build final predictions dict (every S1 entity must appear)
    predictions = {s1_id: sorted(set(s1_to_matches.get(s1_id, []))) for s1_id in all_test_s1_ids}

    matched = sum(1 for v in predictions.values() if v)
    singletons = sum(1 for v in predictions.values() if not v)
    logger.info(f"Entities with ≥1 predicted match: {matched:,}, predicted singletons: {singletons:,}")

    # ── Step 5: Write matching_results.tsv ────────────────────────────────────
    logger.info("Writing matching_results.tsv...")
    write_matching_results(
        str(OUTPUT / "matching_results.tsv"),
        predictions,
        all_test_s1_ids,
    )

    elapsed = time.time() - t_start
    logger.info("=" * 55)
    logger.info(f"Baseline complete in {elapsed/60:.1f} minutes")
    logger.info(f"Output: {OUTPUT}/matching_results.tsv  ←  upload this")
    logger.info(f"        {OUTPUT}/candidate_pairs.tsv")
    logger.info("=" * 55)
    logger.info("Next step:")
    logger.info("  python3 utils/validate_submission.py \\")
    logger.info("    --matching output/matching_results.tsv \\")
    logger.info("    --candidate output/candidate_pairs.tsv \\")
    logger.info("    --test-dir dataset/test")


if __name__ == "__main__":
    main()
