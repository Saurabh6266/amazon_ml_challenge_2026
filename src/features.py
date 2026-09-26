"""
features.py — Phase 2: Pairwise Feature Engineering

Computes string similarity features between S1 and candidate record pairs.
All features are language-agnostic mathematical string distances — no regex,
no country-specific logic. This ensures generalization to unseen French records.

Feature groups:
  1. TF-IDF cosine (from blocking — already computed, free to reuse)
  2. RapidFuzz string distances: Jaro-Winkler, Levenshtein ratio, token ratios
  3. Length difference signals
  4. Multilingual semantic embedding cosine (paraphrase-multilingual-MiniLM-L12-v2)
     → Apache 2.0, 118M params (well under 8B limit), handles French natively

The embedding feature is computed in batch over all pairs for efficiency.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from rapidfuzz import distance as rf_distance
from rapidfuzz import fuzz as rf_fuzz

from normalize import normalize_text

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "tfidf_cosine",
    "name_jaro_winkler",
    "name_levenshtein_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "addr_jaro_winkler",
    "addr_levenshtein_ratio",
    "addr_token_set_ratio",
    "combined_token_set_ratio",
    "name_len_diff_norm",
    "addr_len_diff_norm",
    "name_len_ratio",
    "addr_len_ratio",
    "embedding_cosine",  # added after batch embedding step
]


def pair_features_basic(
    name1: Optional[str],
    addr1: Optional[str],
    name2: Optional[str],
    addr2: Optional[str],
    tfidf_sim: float,
) -> Dict[str, float]:
    """Compute all non-embedding features for one pair.

    Args:
        name1, addr1: S1 business_name and business_address
        name2, addr2: Candidate business_name and business_address
        tfidf_sim: Pre-computed TF-IDF cosine from blocking stage (reused free)

    Returns:
        Dict of feature_name → float value
    """
    n1 = normalize_text(name1)
    n2 = normalize_text(name2)
    a1 = normalize_text(addr1)
    a2 = normalize_text(addr2)
    c1 = (n1 + " " + a1).strip()
    c2 = (n2 + " " + a2).strip()

    # Length-based features (normalized to [0,1] to be scale-invariant)
    len_n1 = max(len(n1), 1)
    len_n2 = max(len(n2), 1)
    len_a1 = max(len(a1), 1)
    len_a2 = max(len(a2), 1)

    return {
        "tfidf_cosine": float(tfidf_sim),
        # Name features
        "name_jaro_winkler": rf_distance.JaroWinkler.similarity(n1, n2),
        "name_levenshtein_ratio": rf_fuzz.ratio(n1, n2) / 100.0,
        "name_token_sort_ratio": rf_fuzz.token_sort_ratio(n1, n2) / 100.0,
        "name_token_set_ratio": rf_fuzz.token_set_ratio(n1, n2) / 100.0,
        # Address features
        "addr_jaro_winkler": rf_distance.JaroWinkler.similarity(a1, a2),
        "addr_levenshtein_ratio": rf_fuzz.ratio(a1, a2) / 100.0,
        "addr_token_set_ratio": rf_fuzz.token_set_ratio(a1, a2) / 100.0,
        # Combined
        "combined_token_set_ratio": rf_fuzz.token_set_ratio(c1, c2) / 100.0,
        # Length signals
        "name_len_diff_norm": abs(len_n1 - len_n2) / max(len_n1, len_n2),
        "addr_len_diff_norm": abs(len_a1 - len_a2) / max(len_a1, len_a2),
        "name_len_ratio": min(len_n1, len_n2) / max(len_n1, len_n2),
        "addr_len_ratio": min(len_a1, len_a2) / max(len_a1, len_a2),
        "embedding_cosine": 0.0,  # placeholder, filled in batch
    }


# ── Multilingual Embedding Model ──────────────────────────────────────────────

_embed_model = None


def _get_embed_model():
    """Lazy-load the multilingual sentence transformer (downloads on first call)."""
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading paraphrase-multilingual-MiniLM-L12-v2 (~118M params, Apache 2.0)...")
        _embed_model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        logger.info("Embedding model loaded.")
    return _embed_model


def embed_texts_batch(texts: List[str], batch_size: int = 512) -> np.ndarray:
    """Encode a list of texts to L2-normalized embeddings.

    Args:
        texts: List of raw (non-normalized) strings — the model handles multilingual
        batch_size: Batch size for GPU/CPU encoding

    Returns:
        np.ndarray of shape (len(texts), embedding_dim), L2-normalized
    """
    model = _get_embed_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embeddings


def compute_embedding_cosines(
    s1_texts: List[str],
    cand_texts: List[str],
) -> np.ndarray:
    """Compute cosine similarities for paired texts in a single batch.

    Both lists must be the same length (aligned pairs).
    Since embeddings are L2-normalized, cosine = dot product.
    """
    all_texts = s1_texts + cand_texts
    all_embs = embed_texts_batch(all_texts)
    s1_embs = all_embs[: len(s1_texts)]
    cand_embs = all_embs[len(s1_texts) :]
    # Dot product of already-normalized vectors = cosine similarity
    cosines = (s1_embs * cand_embs).sum(axis=1)
    return cosines.astype(np.float32)


def build_feature_matrix(
    pairs: List[Tuple[str, str, float]],
    id_to_row: Dict[str, dict],
    use_embeddings: bool = True,
) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
    """Build a feature matrix for a list of candidate pairs.

    Args:
        pairs: [(s1_id, candidate_id, tfidf_sim), ...]
        id_to_row: {entity_id: row_dict}
        use_embeddings: If True, compute multilingual embedding cosine feature

    Returns:
        (X, pair_ids):
          X: np.ndarray of shape (n_pairs, n_features)
          pair_ids: [(s1_id, cand_id), ...] in same row order as X
    """
    if not pairs:
        return np.empty((0, len(FEATURE_COLS)), dtype=np.float32), []

    pair_ids = [(s1_id, cand_id) for s1_id, cand_id, _ in pairs]
    feature_list = []

    for s1_id, cand_id, sim in pairs:
        s1 = id_to_row.get(s1_id, {})
        cand = id_to_row.get(cand_id, {})
        feats = pair_features_basic(
            s1.get("business_name"),
            s1.get("business_address"),
            cand.get("business_name"),
            cand.get("business_address"),
            tfidf_sim=sim,
        )
        feature_list.append(feats)

    # Fill embedding cosines in batch (far more efficient than per-pair)
    if use_embeddings:
        s1_texts = []
        cand_texts = []
        for s1_id, cand_id, _ in pairs:
            s1 = id_to_row.get(s1_id, {})
            cand = id_to_row.get(cand_id, {})
            s1_texts.append(
                (s1.get("business_name") or "") + " " + (s1.get("business_address") or "")
            )
            cand_texts.append(
                (cand.get("business_name") or "") + " " + (cand.get("business_address") or "")
            )
        cosines = compute_embedding_cosines(s1_texts, cand_texts)
        for i, feats in enumerate(feature_list):
            feats["embedding_cosine"] = float(cosines[i])

    X = np.array(
        [[f[col] for col in FEATURE_COLS] for f in feature_list],
        dtype=np.float32,
    )
    return X, pair_ids
