"""Central config for the Creative Similarity Score prototype.

See 04-Technical/2026-07-03-meta-creative-similarity-score-replication.md for the
methodology this implements (replicating Meta's Andromeda / Entity-ID similarity
concept with our own explainable, per-modality embeddings).
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Models (tiny; run on CPU, 16GB Mac is plenty) -----------------------------
VISUAL_MODEL = "clip-ViT-B-32"          # sentence-transformers CLIP: encodes PIL images AND text
TEXT_MODEL = "all-MiniLM-L6-v2"         # sentence-transformers: short-text semantic similarity

# --- Composite scoring ---------------------------------------------------------
# Meta weights visual heaviest; these are tunable in the UI, not gospel.
DEFAULT_VISUAL_WEIGHT = 0.7
DEFAULT_TEXT_WEIGHT = 0.3

# Verdict bands on the composite (0-1). Meta's real thresholds are hidden; these are ours.
VERDICT_BANDS = [
    (0.80, "high-overlap"),   # >= 0.80
    (0.60, "moderate"),       # >= 0.60
    (0.00, "unique"),         # < 0.60
]

# --- Meta Marketing API --------------------------------------------------------
META_API_VERSION = os.getenv("META_API_VERSION", "v21.0")
META_ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN", "").strip()  # needs ads_read
GRAPH_BASE = f"https://graph.facebook.com/{META_API_VERSION}"

# The 4 Cuemath ad accounts (from ads_get_ad_accounts probe, 2026-07-03).
AD_ACCOUNTS = {
    "888586384639855": "Cuemath-Demand-india",
    "925205080936963": "Cuemath-Intel-ROW",
    "5215842511824318": "Cuemath-US & Canada",
    "654380638552528": "Cuemath (unnamed)",
}
DEFAULT_ACCOUNT = "888586384639855"

# --- Local storage (prototype: parquet; prod: Redshift marketing.creative_similarity)
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
EMBED_CACHE = DATA_DIR / "active_embeddings.pkl"       # {account_id: [CreativeRecord,...]}
EMBED_CACHE_VERSION = 2                                 # bump when the Creative schema changes → auto-invalidates stale cache
SCORES_TABLE = DATA_DIR / "creative_similarity.parquet"  # the "separate table"

# How many neighbours to surface per modality in the UI.
TOP_N = 5


def verdict_for(score: float) -> str:
    for threshold, label in VERDICT_BANDS:
        if score >= threshold:
            return label
    return "unique"
