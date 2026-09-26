"""
normalize.py — Language-agnostic text normalization.

Critical design decision: NO country-specific regex or suffix stripping.
We handle US, India (incl. Devanagari), and unseen France equally through:
  1. Unicode NFKD + ASCII transliteration (strips diacritics safely)
  2. Lowercasing
  3. Punctuation → space
  4. Whitespace collapse

Blocking currently uses normalized whole words. RapidFuzz character/string
metrics are applied after candidate generation; typo-only pairs may be missed.
"""

import re
import unicodedata

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_text(s) -> str:
    """Normalize a business name or address to a clean ASCII lowercase string.

    Works identically for US, Indian (including Devanagari transliteration),
    and French text — no language-specific logic.
    """
    if not s:
        return ""
    # NFKD decomposes accented chars (e.g. e-acute -> e + combining accent)
    s = unicodedata.normalize("NFKD", s)
    # Drop all non-ASCII bytes (strips combining diacritics, Devanagari, etc.)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


def char_ngrams(s: str, n: int = 3) -> str:
    """Convert a normalized string to space-joined character n-grams.

    Spaces become underscores so word boundaries create distinct n-grams.
    Used as 'words' for TF-IDF word-level tokenizer to build char-level features.
    """
    s = s.replace(" ", "_")
    if len(s) < n:
        return s
    return " ".join(s[i : i + n] for i in range(len(s) - n + 1))


def build_block_text(name, address) -> str:
    """Combine normalized name+address into TF-IDF-ready word string.

    Whole-word TF-IDF is the current blocking representation. It is sparse but
    does not guarantee O(1) retrieval and may miss typo-only overlaps.
    """
    return (normalize_text(name) + " " + normalize_text(address)).strip()
