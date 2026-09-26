"""
blocking.py — Phase 1: Candidate Generation (Memory-Safe + Ultra-Fast)

Data analysis results:
  - US: 1.3M S1 entities, 6.2M S2+S3 candidates
  - India: 0.88M S1, 5.0M candidates

Memory-safe architecture using sparse_dot_topn (C++ optimized):
  1. Country hard partition (zero recall loss)
  2. TF-IDF vectorized ONCE per country (sparse matrices)
  3. Candidate matrix is transposed (CSC format)
  4. S1 matrix is chunked (e.g. 5,000 rows at a time)
  5. sp_matmul_topn is used to directly compute the top-K sparse dot products
     without EVER materializing a dense intermediate array!
  6. Avoids a dense all-pairs matrix. Runtime still depends on token overlap and
     corpus size; the current chunks can take substantial time.
"""

import gc
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sparse_dot_topn import sp_matmul_topn

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
) -> Dict[str, List[Tuple[str, float]]]:
    """Block one country: TF-IDF fit ONCE, then fast C++ sparse matmul."""
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
        max_df=0.02,  # CRITICAL: drops stop words (e.g. 'inc') to keep sparse matmul fast
        max_features=150_000,
    )
    logger.info(f"    Fitting TF-IDF on {n_s1 + n_cand:,} texts...")
    vec.fit(s1_texts + cand_texts)

    # ── Step 2: Transform S1 and Cands ───────────────────────────────────────
    s1_mat = normalize(vec.transform(s1_texts), norm="l2")  # CSR
    logger.info(f"    S1 matrix: {s1_mat.shape}, nnz={s1_mat.nnz:,}")

    # Build candidates in chunks to avoid memory spikes, then vstack
    # Actually, for 6.2M it's ~4GB which fits in RAM. But let's be safe.
    logger.info(f"    Vectorizing {n_cand:,} candidates...")
    cand_mat = normalize(vec.transform(cand_texts), norm="l2") # CSR
    logger.info(f"    Candidate matrix: {cand_mat.shape}, nnz={cand_mat.nnz:,}")

    # Transpose candidate matrix to CSC for efficient sparse matmul
    logger.info(f"    Transposing candidate matrix...")
    cand_mat_T = cand_mat.T.tocsr() # sparse_dot_topn expects CSR for the second argument

    # ── Step 3: Fast chunked sparse_dot_topn ─────────────────────────────────
    out: Dict[str, List[Tuple[str, float]]] = {}

    s1_chunk_size = 50_000
    n_chunks = (n_s1 + s1_chunk_size - 1) // s1_chunk_size
    logger.info(f"    Starting sparse_dot_topn ({n_chunks} chunks of size {s1_chunk_size:,})...")

    for chunk_idx in range(n_chunks):
        s_start = chunk_idx * s1_chunk_size
        s_end = min(s_start + s1_chunk_size, n_s1)
        s1_slice = s1_mat[s_start:s_end]

        # C-optimized sparse matrix multiplication keeping only top-K elements
        # Returns a CSR matrix
        result = sp_matmul_topn(
            s1_slice, cand_mat_T, top_n=actual_k, threshold=sim_floor, n_threads=8
        )

        # Unpack CSR results
        indptr = result.indptr
        indices = result.indices
        data = result.data

        for local_i in range(s_end - s_start):
            global_i = s_start + local_i
            s1_id = s1_ids[global_i]

            start_ptr = indptr[local_i]
            end_ptr = indptr[local_i + 1]

            pairs = []
            for ptr in range(start_ptr, end_ptr):
                cand_idx = indices[ptr]
                sim = float(data[ptr])
                pairs.append((cand_ids[cand_idx], sim))

            # sort descending
            pairs.sort(key=lambda x: -x[1])
            out[s1_id] = pairs

        if (chunk_idx + 1) % max(1, n_chunks // 5) == 0:
            logger.info(f"    Chunk {chunk_idx+1}/{n_chunks} done")
            gc.collect()

    del s1_mat, cand_mat, cand_mat_T, s1_texts, cand_texts
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
) -> Tuple[Dict[str, List[Tuple[str, float]]], Dict[str, dict]]:
    """Full blocking pipeline using ultra-fast sparse_dot_topn."""
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
