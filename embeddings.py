"""Embedding wrappers — visual (CLIP) + text (MiniLM).

Both models come from sentence-transformers, so there's one dependency and one
cache. Models are lazy-loaded and cached process-wide so Streamlit reruns are cheap.
"""
from __future__ import annotations

import io
from functools import lru_cache
from typing import Optional

import numpy as np
import requests
from PIL import Image

import config


@lru_cache(maxsize=1)
def _visual_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.VISUAL_MODEL)


@lru_cache(maxsize=1)
def _text_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.TEXT_MODEL)


def load_image(source: str | bytes | Image.Image) -> Optional[Image.Image]:
    """Accept a URL, raw bytes, or a PIL image. Returns RGB PIL image or None."""
    try:
        if isinstance(source, Image.Image):
            return source.convert("RGB")
        if isinstance(source, bytes):
            return Image.open(io.BytesIO(source)).convert("RGB")
        if isinstance(source, str) and source:
            # Meta image_url are signed CDN URLs — embed promptly, don't persist the URL.
            resp = requests.get(source, timeout=20)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception as exc:  # noqa: BLE001 - graceful skip on any load failure
        print(f"[embeddings] image load failed for {str(source)[:80]!r}: {exc}")
    return None


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


def _mean_normalize(vecs: list[np.ndarray]) -> Optional[np.ndarray]:
    """Mean-pool a set of unit vectors, then re-normalize so cosine == dot still holds."""
    if not vecs:
        return None
    m = np.mean(np.stack(vecs), axis=0)
    n = np.linalg.norm(m)
    return (m / n).astype(np.float32) if n > 0 else None


def embed_images(sources: list) -> Optional[np.ndarray]:
    """Embed several image sources (URLs/bytes/PIL) and mean-pool.

    For dynamic creatives this pools all static images + video thumbnails into one
    'visual centroid' for the creative. Failed/undownloadable sources are skipped.
    """
    vecs = [v for s in (sources or []) if (v := embed_image(s)) is not None]
    return _mean_normalize(vecs)


def embed_texts(texts: list) -> Optional[np.ndarray]:
    """Embed several copy variants (bodies/titles) and mean-pool into a 'messaging centroid'."""
    vecs = []
    for t in texts or []:
        v = embed_text(body=t)
        if v is not None:
            vecs.append(v)
    return _mean_normalize(vecs)


def cosine(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    """Cosine similarity for already-normalized vectors (dot product)."""
    if a is None or b is None:
        return None
    return float(np.dot(a, b))
