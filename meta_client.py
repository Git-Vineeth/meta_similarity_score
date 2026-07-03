"""Pull ACTIVE creatives from the Meta Marketing API.

Production uses a direct Marketing API token with `ads_read` (the meta-ads MCP is
interactive-only and can't be driven from a standalone script). If META_ACCESS_TOKEN
is unset, we fall back to a small mock set so the UI runs end-to-end immediately.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import requests

import config


@dataclass
class Creative:
    creative_id: str
    name: str
    ad_account_id: str
    title: str = ""          # headline (Ads Manager) / API `title`
    body: str = ""           # primary text (Ads Manager) / API `body`
    image_url: str = ""
    thumbnail_url: str = ""
    video_id: str = ""
    is_active: bool = True
    # filled in by the scorer:
    visual_vec: Optional[object] = field(default=None, repr=False)
    copy_vec: Optional[object] = field(default=None, repr=False)

    @property
    def best_image(self) -> str:
        return self.image_url or self.thumbnail_url


def fetch_active_creatives(ad_account_id: str, limit: int = 200) -> list[Creative]:
    """Return ACTIVE ads' creatives for an account. Mock fallback when no token."""
    if not config.META_ACCESS_TOKEN:
        return _mock_creatives(ad_account_id)

    url = f"{config.GRAPH_BASE}/act_{ad_account_id}/ads"
    params = {
        "fields": "name,effective_status,"
        "creative{id,name,status,body,title,image_url,thumbnail_url,video_id}",
        "effective_status": '["ACTIVE"]',
        "limit": min(limit, 100),
        "access_token": config.META_ACCESS_TOKEN,
    }
    out: list[Creative] = []
    while url:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        for ad in payload.get("data", []):
            cr = ad.get("creative") or {}
            if not cr.get("id"):
                continue
            out.append(
                Creative(
                    creative_id=str(cr["id"]),
                    name=cr.get("name") or ad.get("name", ""),
                    ad_account_id=ad_account_id,
                    title=cr.get("title", "") or "",
                    body=cr.get("body", "") or "",
                    image_url=cr.get("image_url", "") or "",
                    thumbnail_url=cr.get("thumbnail_url", "") or "",
                    video_id=str(cr.get("video_id", "") or ""),
                    is_active=True,
                )
            )
            if len(out) >= limit:
                return out
        url = payload.get("paging", {}).get("next")
        params = {}  # `next` is a fully-formed URL
    return out


def _mock_creatives(ad_account_id: str) -> list[Creative]:
    """A tiny stand-in active set so the app runs before the API token is wired.
    Two of these are deliberately near-duplicates to exercise the high-overlap path.
    """
    samples = [
        ("m1", "Live 1:1 Math Tutoring",
         "Book a FREE trial class today! Personalised 1-on-1 online math tutoring for grades 1-10.",
         "https://picsum.photos/seed/cuemath1/512/512"),
        ("m2", "Free Math Trial Class",
         "Start with a FREE trial! One-on-one online maths classes for your child, grades 1 to 10.",
         "https://picsum.photos/seed/cuemath1/512/512"),  # near-dup of m1 (same image seed + similar copy)
        ("m3", "Build Math Confidence",
         "Help your child fall in love with numbers. Certified tutors, live classes, real results.",
         "https://picsum.photos/seed/cuemath2/512/512"),
        ("m4", "Coding for Kids",
         "Beyond math — now teaching coding & logic for young learners. Try a free session.",
         "https://picsum.photos/seed/cuemath3/512/512"),
    ]
    return [
        Creative(
            creative_id=f"{cid}",
            name=name,
            ad_account_id=ad_account_id,
            title=name,
            body=body,
            image_url=img,
            is_active=True,
        )
        for cid, name, body, img in samples
    ]


def using_mock() -> bool:
    return not config.META_ACCESS_TOKEN
