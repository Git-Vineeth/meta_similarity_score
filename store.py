"""Persistence for the prototype.

- Active-set embeddings are cached to a pickle so re-scoring a new candidate is a cheap
  vector lookup (no re-embedding the whole account).
- Scoring results are appended to a parquet file that mirrors the prod
  Redshift table `marketing.creative_similarity`.
"""
from __future__ import annotations

import pickle
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

import config
from meta_client import Creative
from similarity import ScoreResult

# Column order matches the DDL in the design note (marketing.creative_similarity).
SCORE_COLUMNS = [
    "scored_at", "ad_account_id", "creative_id", "submission_id", "is_active",
    "visual_sim_max", "visual_sim_mean", "copy_sim_max", "copy_sim_mean",
    "similarity_score", "nearest_creative_id", "nearest_modality", "verdict",
]


# --- embedding cache -----------------------------------------------------------
def save_embeddings(account_id: str, creatives: list[Creative]) -> None:
    data = _load_all_embeddings()
    data[account_id] = creatives
    envelope = {"__version__": config.EMBED_CACHE_VERSION, "data": data}
    with open(config.EMBED_CACHE, "wb") as fh:
        pickle.dump(envelope, fh)


def load_embeddings(account_id: str) -> Optional[list[Creative]]:
    return _load_all_embeddings().get(account_id)


def _load_all_embeddings() -> dict[str, list[Creative]]:
    """Load the cache, but treat a version mismatch (old Creative schema) as empty
    so it rebuilds instead of unpickling stale objects that lack new attributes."""
    if config.EMBED_CACHE.exists():
        try:
            with open(config.EMBED_CACHE, "rb") as fh:
                blob = pickle.load(fh)
            if isinstance(blob, dict) and blob.get("__version__") == config.EMBED_CACHE_VERSION:
                return blob.get("data", {})
        except Exception:  # noqa: BLE001 - stale/corrupt cache → rebuild
            pass
    return {}


# --- scores table --------------------------------------------------------------
def append_score(
    account_id: str,
    result: ScoreResult,
    *,
    creative_id: Optional[str] = None,
    submission_id: Optional[str] = None,
    is_active: bool = False,
) -> None:
    row = {
        "scored_at": datetime.now(timezone.utc),
        "ad_account_id": account_id,
        "creative_id": creative_id,
        "submission_id": submission_id,
        "is_active": is_active,
        "visual_sim_max": result.visual_sim_max,
        "visual_sim_mean": result.visual_sim_mean,
        "copy_sim_max": result.copy_sim_max,
        "copy_sim_mean": result.copy_sim_mean,
        "similarity_score": result.similarity_score,
        "nearest_creative_id": result.nearest_creative_id,
        "nearest_modality": result.nearest_modality,
        "verdict": result.verdict,
    }
    df_new = pd.DataFrame([row], columns=SCORE_COLUMNS)
    if config.SCORES_TABLE.exists():
        df = pd.concat([pd.read_parquet(config.SCORES_TABLE), df_new], ignore_index=True)
    else:
        df = df_new
    df.to_parquet(config.SCORES_TABLE, index=False)


def read_scores() -> pd.DataFrame:
    if config.SCORES_TABLE.exists():
        return pd.read_parquet(config.SCORES_TABLE)
    return pd.DataFrame(columns=SCORE_COLUMNS)
