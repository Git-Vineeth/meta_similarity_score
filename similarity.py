"""Scoring: candidate creative vs the active set → per-modality + composite score.

Mirrors Meta's cannibalization framing (a creative's score = how close it is to the
most-similar OTHER live creative), but keeps visual and copy separate for explainability.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

import config
from embeddings import cosine, embed_image, embed_text
from meta_client import Creative


@dataclass
class Neighbor:
    creative_id: str
    name: str
    similarity: float
    modality: str          # 'visual' | 'copy'
    image_url: str = ""
    title: str = ""
    body: str = ""


@dataclass
class ScoreResult:
    visual_sim_max: Optional[float]
    visual_sim_mean: Optional[float]
    copy_sim_max: Optional[float]
    copy_sim_mean: Optional[float]
    similarity_score: float                 # 0-100 composite
    verdict: str
    nearest_creative_id: Optional[str]
    nearest_modality: Optional[str]
    visual_neighbors: list[Neighbor]
    copy_neighbors: list[Neighbor]


def ensure_embeddings(creatives: list[Creative]) -> list[Creative]:
    """Populate visual_vec / copy_vec in place (idempotent)."""
    for c in creatives:
        if c.visual_vec is None and c.best_image:
            c.visual_vec = embed_image(c.best_image)
        if c.copy_vec is None:
            c.copy_vec = embed_text(c.title, c.body)
    return creatives


def _stats(sims: list[float]) -> tuple[Optional[float], Optional[float]]:
    if not sims:
        return None, None
    return max(sims), float(np.mean(sims))


def score_candidate(
    cand_visual: Optional[np.ndarray],
    cand_copy: Optional[np.ndarray],
    active: list[Creative],
    *,
    exclude_id: Optional[str] = None,
    w_visual: float = config.DEFAULT_VISUAL_WEIGHT,
    w_text: float = config.DEFAULT_TEXT_WEIGHT,
    top_n: int = config.TOP_N,
) -> ScoreResult:
    visual_hits: list[Neighbor] = []
    copy_hits: list[Neighbor] = []

    for c in active:
        if exclude_id is not None and c.creative_id == exclude_id:
            continue
        vs = cosine(cand_visual, c.visual_vec)
        if vs is not None:
            visual_hits.append(Neighbor(c.creative_id, c.name, vs, "visual",
                                        image_url=c.best_image, title=c.title, body=c.body))
        cs = cosine(cand_copy, c.copy_vec)
        if cs is not None:
            copy_hits.append(Neighbor(c.creative_id, c.name, cs, "copy",
                                      image_url=c.best_image, title=c.title, body=c.body))

    visual_hits.sort(key=lambda n: n.similarity, reverse=True)
    copy_hits.sort(key=lambda n: n.similarity, reverse=True)

    v_max, v_mean = _stats([n.similarity for n in visual_hits])
    c_max, c_mean = _stats([n.similarity for n in copy_hits])

    # Composite: weight the two max-similarities. Renormalize when a modality is absent
    # (e.g. no image on the candidate) so we don't unfairly deflate the score.
    parts, weights = [], []
    if v_max is not None:
        parts.append(v_max); weights.append(w_visual)
    if c_max is not None:
        parts.append(c_max); weights.append(w_text)
    composite = float(np.average(parts, weights=weights)) if parts else 0.0

    # Nearest neighbour = whichever modality is closest overall (drives the callout).
    nearest_id, nearest_mod = None, None
    best = -1.0
    if visual_hits and visual_hits[0].similarity > best:
        best, nearest_id, nearest_mod = visual_hits[0].similarity, visual_hits[0].creative_id, "visual"
    if copy_hits and copy_hits[0].similarity > best:
        best, nearest_id, nearest_mod = copy_hits[0].similarity, copy_hits[0].creative_id, "copy"

    return ScoreResult(
        visual_sim_max=v_max, visual_sim_mean=v_mean,
        copy_sim_max=c_max, copy_sim_mean=c_mean,
        similarity_score=round(composite * 100, 1),
        verdict=config.verdict_for(composite),
        nearest_creative_id=nearest_id, nearest_modality=nearest_mod,
        visual_neighbors=visual_hits[:top_n], copy_neighbors=copy_hits[:top_n],
    )
