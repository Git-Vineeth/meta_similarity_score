"""Embedding wrappers — visual (CLIP) + text (MiniLM).

Both models come from sentence-transformers (one dependency, one cache) and are
lazy-loaded + process-cached so Streamlit reruns are cheap.

Batched, parallel path (`embed_image_groups` / `embed_text_groups`) is what the
active-set index uses: it downloads images concurrently, dedupes repeated URLs,
and encodes in batches — orders of magnitude faster than one-at-a-time for large
accounts (e.g. US, ~130 creatives × several images each).
"""
from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Callable, Optional

import numpy as np
import requests
from PIL import Image

import config

Progress = Optional[Callable[[float, str], None]]


@lru_cache(maxsize=1)
def _visual_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.VISUAL_MODEL)


@lru_cache(maxsize=1)
def _text_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.TEXT_MODEL)


def _report(progress: Progress, frac: float, msg: str) -> None:
    if progress:
        try:
            progress(max(0.0, min(frac, 1.0)), msg)
        except Exception:  # noqa: BLE001 - never let a UI callback break embedding
            pass


# ------------------------------------------------------------------ loading ---
def load_image(source: str | bytes | Image.Image) -> Optional[Image.Image]:
    """Accept a URL, raw bytes, or a PIL image. Returns RGB PIL image or None."""
    try:
        if isinstance(source, Image.Image):
            return source.convert("RGB")
        if isinstance(source, bytes):
            return Image.open(io.BytesIO(source)).convert("RGB")
        if isinstance(source, str) and source:
            # Meta image_url are signed CDN URLs — embed promptly, don't persist the URL.
            resp = requests.get(source, timeout=config.IMAGE_TIMEOUT)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception as exc:  # noqa: BLE001 - graceful skip on any load failure
        print(f"[embeddings] image load failed for {str(source)[:80]!r}: {exc}")
    return None


def _mean_normalize(vecs: list[np.ndarray]) -> Optional[np.ndarray]:
    """Mean-pool unit vectors then re-normalize so cosine == dot still holds."""
    if not vecs:
        return None
    m = np.mean(np.stack(vecs), axis=0)
    n = np.linalg.norm(m)
    return (m / n).astype(np.float32) if n > 0 else None


# ----------------------------------------------------- single (candidate) -----
def embed_image(source: str | bytes | Image.Image | None) -> Optional[np.ndarray]:
    img = load_image(source) if source is not None else None
    if img is None:
        return None
    vec = _visual_model().encode(img, normalize_embeddings=True)
    return np.asarray(vec, dtype=np.float32)


def embed_text(title: str = "", body: str = "") -> Optional[np.ndarray]:
    text = " ".join(p for p in (title or "", body or "") if p).strip()
    if not text:
        return None
    vec = _text_model().encode(text, normalize_embeddings=True)
    return np.asarray(vec, dtype=np.float32)


def embed_images(sources: list) -> Optional[np.ndarray]:
    """Mean-pool several image sources into one visual centroid (candidate use)."""
    return embed_image_groups([list(sources or [])])[0]


def embed_texts(texts: list) -> Optional[np.ndarray]:
    """Mean-pool several copy variants into one messaging centroid (candidate use)."""
    return embed_text_groups([list(texts or [])])[0]


# -------------------------------------------------- batched (active set) ------
def embed_image_groups(groups: list[list], progress: Progress = None) -> list[Optional[np.ndarray]]:
    """One visual centroid per group. Downloads unique URLs in parallel, batch-encodes."""
    groups = [list(g or [])[: config.MAX_IMAGES_PER_CREATIVE] for g in groups]

    # unique string URLs → download once (bytes/PIL sources handled inline, rare here)
    unique_urls = list({s for g in groups for s in g if isinstance(s, str)})
    url_to_img: dict[str, Optional[Image.Image]] = {}
    if unique_urls:
        done = 0
        with ThreadPoolExecutor(max_workers=config.DOWNLOAD_WORKERS) as pool:
            for url, img in zip(unique_urls, pool.map(load_image, unique_urls)):
                url_to_img[url] = img
                done += 1
                _report(progress, 0.6 * done / len(unique_urls),
                        f"Downloading images {done}/{len(unique_urls)}…")

    # collect all PIL images with their group index, then batch-encode once
    imgs, owners = [], []
    for gi, g in enumerate(groups):
        for s in g:
            img = url_to_img.get(s) if isinstance(s, str) else load_image(s)
            if img is not None:
                imgs.append(img)
                owners.append(gi)

    _report(progress, 0.65, f"Encoding {len(imgs)} images…")
    vecs = _encode_batch(_visual_model(), imgs) if imgs else []

    # mean-pool per group
    per_group: list[list[np.ndarray]] = [[] for _ in groups]
    for owner, vec in zip(owners, vecs):
        per_group[owner].append(vec)
    return [_mean_normalize(v) for v in per_group]


def embed_text_groups(groups: list[list], progress: Progress = None) -> list[Optional[np.ndarray]]:
    """One copy centroid per group. Batch-encodes all variants at once."""
    groups = [[t for t in (g or []) if isinstance(t, str) and t.strip()] for g in groups]
    texts, owners = [], []
    for gi, g in enumerate(groups):
        for t in g:
            texts.append(t)
            owners.append(gi)

    _report(progress, 0.9, f"Encoding {len(texts)} copy variants…")
    vecs = _encode_batch(_text_model(), texts) if texts else []

    per_group: list[list[np.ndarray]] = [[] for _ in groups]
    for owner, vec in zip(owners, vecs):
        per_group[owner].append(vec)
    return [_mean_normalize(v) for v in per_group]


def _encode_batch(model, items: list) -> list[np.ndarray]:
    arr = model.encode(items, batch_size=config.EMBED_BATCH,
                       normalize_embeddings=True, convert_to_numpy=True,
                       show_progress_bar=False)
    return [np.asarray(v, dtype=np.float32) for v in arr]


def cosine(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    """Cosine similarity for already-normalized vectors (dot product)."""
    if a is None or b is None:
        return None
    return float(np.dot(a, b))
