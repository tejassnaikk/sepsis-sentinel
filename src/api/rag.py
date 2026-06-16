"""
src/api/rag.py

RAG retrieval for SepsisSentinel.
Given a stay_id, retrieves relevant passages from the patient's discharge
summary that explain or contextualize a sepsis alert.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

# Lazy imports — only loaded when RAG is first called
_embedder = None
_discharge: Optional[pd.DataFrame] = None
_icustays:  Optional[pd.DataFrame] = None

DATA_DIR = Path("/Volumes/Tejas SSD/sepsis-sentinel/data/raw")
DISCHARGE_PATH = DATA_DIR / "discharge.csv.gz"
ICUSTAYS_PATH  = DATA_DIR / "icustays.csv.gz"

SEPSIS_QUERY = (
    "sepsis infection fever hypotension organ dysfunction "
    "lactate blood culture antibiotic treatment"
)


def _load_resources() -> None:
    """Lazy-load the embedder and data files on first call."""
    global _embedder, _discharge, _icustays
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    if _discharge is None:
        _discharge = pd.read_csv(
            str(DISCHARGE_PATH),
            usecols=["hadm_id", "text"],
        )
    if _icustays is None:
        _icustays = pd.read_csv(
            str(ICUSTAYS_PATH),
            usecols=["stay_id", "hadm_id"],
        )


def _chunk_text(
    text: str,
    chunk_words: int = 200,
    stride: int = 100,
    min_words: int = 30,
) -> list[str]:
    """Split text into overlapping word-level chunks."""
    words = text.split()
    chunks: list[str] = []
    for i in range(0, len(words), stride):
        chunk = " ".join(words[i : i + chunk_words])
        if len(chunk.split()) >= min_words:
            chunks.append(chunk)
        if i + chunk_words >= len(words):
            break
    return chunks


def retrieve(
    stay_id: int,
    top_k: int = 3,
    query: str = SEPSIS_QUERY,
) -> list[dict]:
    """
    Retrieve the top-k most relevant discharge note passages for a stay.

    Parameters
    ----------
    stay_id : ICU stay identifier
    top_k   : Number of passages to return
    query   : Retrieval query string

    Returns
    -------
    List of dicts with keys: chunk (str), score (float)
    Empty list if no discharge note found for this stay.
    """
    try:
        _load_resources()

        # stay_id → hadm_id
        hadm_rows = _icustays[_icustays["stay_id"] == stay_id]["hadm_id"].values
        if len(hadm_rows) == 0:
            return []
        hadm_id = int(hadm_rows[0])

        # hadm_id → discharge note text
        note_rows = _discharge[_discharge["hadm_id"] == hadm_id]
        if note_rows.empty:
            return []
        note_text = note_rows.iloc[0]["text"]
        if not isinstance(note_text, str) or len(note_text) < 100:
            return []

        # Chunk and embed
        chunks = _chunk_text(note_text)
        if not chunks:
            return []

        chunk_embeddings = _embedder.encode(chunks, show_progress_bar=False)
        query_embedding  = _embedder.encode([query], show_progress_bar=False)[0]

        # Cosine similarity
        norms = np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
        chunk_embeddings_norm = chunk_embeddings / (norms + 1e-9)
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
        scores = chunk_embeddings_norm @ query_norm

        # Top-k
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [
            {
                "chunk": chunks[i][:500],  # truncate to 500 chars for API response
                "score": round(float(scores[i]), 4),
            }
            for i in top_idx
        ]

    except Exception as exc:
        # RAG failures should never break the prediction response
        return [{"chunk": f"RAG error: {exc}", "score": 0.0}]
