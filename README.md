# Creative Similarity Score — Prototype

Our replication of Meta's **Creative Similarity Score** (the Andromeda / Creative-Entity-ID
concept). Meta's exact formula is a black box, so we reproduce the *methodology*:
embed each creative's **visual** (CLIP) and **copy** (sentence-transformers) content, then
score a creative by its cosine similarity to the most-similar *other* active creative.
Unlike Meta's single opaque number, ours is **explainable** and splits **visual vs copy**.

Full design + rationale lives in the internal knowledge vault:
`04-Technical/2026-07-03-meta-creative-similarity-score-replication.md`

## What it does
- **Pre-flight** — paste a new creative's headline/body + image → get its similarity score
  (0–100), verdict, and the top matching active creatives (per modality) *before* you launch it.
- **Monitoring** — scan the whole active set to flag creatives that cannibalize each other.
- Persists every score to `data/creative_similarity.parquet` — the prototype stand-in for the
  prod Redshift table `marketing.creative_similarity`.

## Setup (Python 3.11 venv recommended)
```bash
cd creative-similarity-prototype
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # optional: add META_ACCESS_TOKEN for live data
streamlit run app.py
```
First run downloads the two models (~700MB total, CPU-only, fine on a 16GB Mac).
**Without a token it runs on a mock active set** (two are near-duplicates, so you can see the
high-overlap path immediately).

## Files
| File | Role |
|---|---|
| `app.py` | Streamlit UI (pre-flight / monitoring / history tabs) |
| `embeddings.py` | CLIP (visual) + MiniLM (copy) wrappers + cosine |
| `meta_client.py` | Pull ACTIVE creatives from Marketing API (+ mock fallback) |
| `similarity.py` | Per-modality + composite scoring, nearest neighbour |
| `store.py` | Embedding cache + `creative_similarity` parquet table |
| `config.py` | Models, weights, thresholds, the 4 ad accounts |

## Known limits (prototype)
- **Video**: not yet embedded — image + copy only. v1.1 adds ffmpeg/opencv keyframe extraction
  (mean-pooled CLIP), per the design note.
- **Weights** (default 70/30 visual/copy) and verdict thresholds (`0.60` / `0.80`) are tunable in
  the sidebar / `config.py` — calibrate against known-similar Cuemath creatives.
- **Live mode** needs a Marketing API `ads_read` token (the MCP can't drive a standalone script).
- Meta `image_url`s are signed and expire — we embed on fetch, never persist the URL as truth.
