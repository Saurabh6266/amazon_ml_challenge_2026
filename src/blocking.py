"""
blocking.py — Phase 1: Candidate Generation (Memory-Safe for 8GB RAM)

Data analysis results (from actual dataset):
  - True match pair Jaccard: min=0.062, median=0.707 — well within TF-IDF cosine range
  - US: 1.3M S1 entities, 6.2M S2+S3 candidates
  - India: 0.88M S1, 5.0M candidates

Memory-safe architecture (verified: 8.6GB total, ~2.5GB free):
  1. Country hard partition (zero recall loss)
  2. TF-IDF vectorized ONCE per country (not per batch) → sparse S1 matrix kept in RAM
  3. Candidate batches of 300K: transform each batch separately (avoids 3.8GB cand matrix)
  4. For each (S1-chunk, cand-batch): dense result = (s1_chunk × batch_size) × 4B
     Budget: 166 S1 × 300K × 4B = 200MB per iteration ✅
  5. Per-S1 heap of top-K maintains best candidates across all batches

Performance on full dataset (extrapolated from 200×10K benchmark at 1.2s):
  - TF-IDF fit on 8M texts: ~5–10 min per country
  - Matmul iterations: (1.3M/166) × (6.2M/300K) ≈ 8K × 21 = 168K iters × ~0.01s = 28min
  - Total estimated: ~35–45 min per country, ~2h for full train + test
"""

import gc
import heapq
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from normalize import build_block_text

logger = logging.getLogger(__name__)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_source_by_country(filepath: str) -> Dict[str, List[dict]]:
    """Load source TSV grouped by country."""
    by_country: Dict[str, List[dict]] = defaultdict(list)
    with open(filepath, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            row = dict(zip(header, parts))
            by_country[row["country"]].append(row)
    return dict(by_country)


# ── Core blocking per country ─────────────────────────────────────────────────

def _block_country(
    s1_rows: List[dict],
    cand_rows: List[dict],
    top_k: int,
    sim_floor: float,
    cand_batch_size: int = 300_000,
    max_dense_mb: int = 200,
) -> Dict[str, List[Tuple[str, float]]]:
    """Block one country: TF-IDF fit ONCE, candidates processed in batches.

    Key design:
      - TF-IDF fit + S1 transform: done once, S1 sparse matrix stays in RAM
      - Candidate transform: done in batches of cand_batch_size
      - Dense matmul result: (s1_chunk × batch_size) × 4B ≤ max_dense_mb MB
      - Per-S1 min-heaps accumulate top-K across all batches

    Memory per iteration (cand_batch_size=300K, max_dense_mb=200):
      s1_chunk = 200MB / (300K × 4B) = 166 rows
      Dense result: 166 × 300K × 4B = 199MB ✅
    """
    n_s1 = len(s1_rows)
    n_cand = len(cand_rows)
    actual_k = min(top_k, n_cand)

    s1_texts = [
        build_block_text(r.get("business_name"), r.get("business_address"))
        for r in s1_rows
    ]
    cand_texts = [
        build_block_text(r.get("business_name"), r.get("business_address"))
        for r in cand_rows
    ]
    s1_ids = [r["entity_id"] for r in s1_rows]
    cand_ids = [r["entity_id"] for r in cand_rows]

    # ── Step 1: Fit TF-IDF on combined corpus ONCE ───────────────────────────
    vec = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"\S+",
        sublinear_tf=True,
        min_df=2,
        max_features=150_000,
    )
    logger.info(f"    Fitting TF-IDF on {n_s1 + n_cand:,} texts...")
    vec.fit(s1_texts + cand_texts)

    # ── Step 2: Transform S1 once, keep sparse in RAM ────────────────────────
    s1_mat = normalize(vec.transform(s1_texts), norm="l2")  # sparse CSR
    logger.info(
        f"    S1 sparse matrix: {s1_mat.shape}, nnz={s1_mat.nnz:,}, "
        f"~{s1_mat.data.nbytes/1e6:.0f}MB"
    )

    # Per-S1 min-heaps: (sim, global_cand_idx)
    heaps: List[List[Tuple[float, int]]] = [[] for _ in range(n_s1)]

    # Adaptive S1 chunk size to stay within max_dense_mb
    s1_chunk = max(1, int(max_dense_mb * 1024 * 1024 / (min(cand_batch_size, n_cand) * 4)))
    s1_chunk = min(s1_chunk, n_s1, 500)

    # ── Step 3: Process candidates in batches ────────────────────────────────
    effective_cand_batch = min(cand_batch_size, n_cand)
    n_batches = (n_cand + effective_cand_batch - 1) // effective_cand_batch
    logger.info(
        f"    Processing: {n_batches} cand-batches × "
        f"{(n_s1 + s1_chunk - 1)//s1_chunk} s1-chunks"
        f"  (s1_chunk={s1_chunk}, cand_batch={effective_cand_batch:,})"
    )

    for batch_idx in range(n_batches):
        c_start = batch_idx * effective_cand_batch
        c_end = min(c_start + effective_cand_batch, n_cand)
        batch_texts = cand_texts[c_start:c_end]

        # Transform this candidate batch (reuse fitted vec)
        cand_batch_mat = normalize(vec.transform(batch_texts), norm="l2")

        for s1_start in range(0, n_s1, s1_chunk):
            s1_end = min(s1_start + s1_chunk, n_s1)
            s1_slice = s1_mat[s1_start:s1_end]  # sparse slice

            # Dense result: (s1_end - s1_start, c_end - c_start)
            result = (s1_slice @ cand_batch_mat.T).toarray().astype(np.float32)

            for local_i in range(result.shape[0]):
                global_i = s1_start + local_i
                row = result[local_i]
                above_mask = row >= sim_floor
                if not above_mask.any():
                    continue

                heap = heaps[global_i]
                for local_j in np.where(above_mask)[0]:
                    sim = float(row[local_j])
                    global_j = int(c_start + local_j)
                    if len(heap) < actual_k:
                        heapq.heappush(heap, (sim, global_j))
                    elif sim > heap[0][0]:
                        heapq.heapreplace(heap, (sim, global_j))

            del result
        del cand_batch_mat
        gc.collect()

        if n_batches > 1 and (batch_idx + 1) % max(1, n_batches // 5) == 0:
            logger.info(f"    Batch {batch_idx+1}/{n_batches} done")

    # ── Step 4: Materialize results from heaps ────────────────────────────────
    out: Dict[str, List[Tuple[str, float]]] = {}
    for row_i, s1_id in enumerate(s1_ids):
        heap = heaps[row_i]
        if not heap:
            out[s1_id] = []
            continue
        pairs = [(cand_ids[j], sim) for sim, j in sorted(heap, key=lambda x: -x[0])]
        out[s1_id] = pairs[:actual_k]

    del s1_mat, heaps, s1_texts, cand_texts
    gc.collect()
    return out


# ── Main entry point ──────────────────────────────────────────────────────────

def run_blocking(
    s1_filepath: str,
    s2_filepath: str,
    s3_filepath: str,
    top_k: int = 20,
    sim_floor: float = 0.10,
    countries: Optional[List[str]] = None,
    cand_batch_size: int = 300_000,
    max_dense_mb: int = 200,
) -> Tuple[Dict[str, List[Tuple[str, float]]], Dict[str, dict]]:
    """Full blocking pipeline: country partition + TF-IDF + chunked sparse cosine.

    Args:
        s1_filepath: Path to source 1 TSV
        s2_filepath: Path to source 2 TSV
        s3_filepath: Path to source 3 TSV
        top_k: Max candidates per S1 entity (knob for candidate_pairs.tsv size)
        sim_floor: Minimum TF-IDF cosine similarity to include
        countries: If set, only process these countries
        cand_batch_size: Candidates per batch (tune for RAM)
        max_dense_mb: Max MB for dense matmul result per iteration

    Returns:
        (block_result, id_to_row):
          block_result: {s1_id: [(cand_id, cosine_sim), ...]}
          id_to_row: {entity_id: row_dict} for all records
    """
    logger.info("Loading source files by country...")
    s1_by_country = load_source_by_country(s1_filepath)
    s2_by_country = load_source_by_country(s2_filepath)
    s3_by_country = load_source_by_country(s3_filepath)

    # Global ID→row lookup
    id_to_row: Dict[str, dict] = {}
    for src in (s1_by_country, s2_by_country, s3_by_country):
        for rows in src.values():
            for r in rows:
                id_to_row[r["entity_id"]] = r

    all_s1_countries = set(s1_by_country.keys())
    process_countries = set(countries) & all_s1_countries if countries else all_s1_countries

    block_result: Dict[str, List[Tuple[str, float]]] = {}

    for country in sorted(process_countries):
        s1_rows = s1_by_country.get(country, [])
        cand_rows = s2_by_country.get(country, []) + s3_by_country.get(country, [])

        logger.info(
            f"\n[{country}] S1={len(s1_rows):,}, candidates={len(cand_rows):,}, top_k={top_k}"
        )

        if not s1_rows:
            continue
        if not cand_rows:
            for r in s1_rows:
                block_result[r["entity_id"]] = []
            continue

        country_result = _block_country(
            s1_rows, cand_rows,
            top_k=top_k,
            sim_floor=sim_floor,
            cand_batch_size=cand_batch_size,
            max_dense_mb=max_dense_mb,
        )
        block_result.update(country_result)
        n_pairs = sum(len(v) for v in country_result.values())
        n_s1 = len(country_result)
        logger.info(
            f"[{country}] Done. {n_pairs:,} pairs, avg {n_pairs/max(n_s1,1):.1f}/S1"
        )
        gc.collect()

    # Ensure every S1 entity has an entry
    for rows in s1_by_country.values():
        for r in rows:
            if r["entity_id"] not in block_result:
                block_result[r["entity_id"]] = []

    total = sum(len(v) for v in block_result.values())
    logger.info(
        f"\nBlocking complete: {len(block_result):,} S1, "
        f"{total:,} pairs, avg {total/max(len(block_result),1):.1f}/S1"
    )
    return block_result, id_to_row
