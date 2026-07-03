"""Pull ACTIVE creatives from the Meta Marketing API and normalize them.

Cuemath's live ads are mostly **Advantage+ dynamic creatives**: the copy and media
live inside `asset_feed_spec` (bodies[]/titles[]/videos[]/images[]), NOT the flat
body/title/image_url fields. This module flattens both dynamic (`asset_feed_spec`)
and classic (`object_story_spec`) creatives into a common shape:
  - texts:         all copy variants (bodies + titles + descriptions)
  - image_sources: all embeddable image URLs (static images + video thumbnails)

Production uses a direct Marketing API token with `ads_read` (the meta-ads MCP is
interactive-only). Without a token we fall back to a small mock set.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import requests

import config


class MetaAPIError(Exception):
    """Clean, token-free Meta Graph API error for the UI to display."""


def _raise_for_meta(resp) -> None:
    """Raise a clean MetaAPIError on a non-OK response — never leak the token/URL."""
    if resp.ok:
        return
    try:
        err = resp.json().get("error", {})
    except Exception:  # noqa: BLE001
        err = {}
    code, msg = err.get("code"), (err.get("message") or "")[:200]
    if code == 190:
        raise MetaAPIError(
            "Meta access token expired or invalid — generate a fresh one and update "
            "META_ACCESS_TOKEN in .env. (Explorer tokens are short-lived; extend via the "
            "Access Token Debugger or use a System User token.)"
        )
    raise MetaAPIError(f"Meta API error {resp.status_code} (code {code}): {msg}")


# Creative fields we request; asset_feed_spec/object_story_spec carry the real payload.
_CREATIVE_FIELDS = (
    "id,name,object_type,body,title,image_url,thumbnail_url,video_id,"
    "object_story_spec,asset_feed_spec"
)


@dataclass
class Creative:
    creative_id: str
    name: str
    ad_account_id: str
    title: str = ""                       # representative headline (for display)
    body: str = ""                        # representative primary text (for display)
    texts: list[str] = field(default_factory=list)          # all copy variants → embed & mean-pool
    image_sources: list[str] = field(default_factory=list)  # all image URLs (incl. video keyframes) → embed & mean-pool
    video_ids: list[str] = field(default_factory=list)      # for keyframe enrichment
    is_active: bool = True
    # filled in by the scorer:
    visual_vec: Optional[object] = field(default=None, repr=False)
    copy_vec: Optional[object] = field(default=None, repr=False)

    @property
    def best_image(self) -> str:
        return self.image_sources[0] if self.image_sources else ""

    @property
    def n_variants(self) -> str:
        return f"{len(self.texts)} copy · {len(self.image_sources)} img"


# ------------------------------------------------------------------ parsing ---
def _texts_from_feed(feed: dict) -> list[str]:
    out = []
    for key in ("bodies", "titles", "descriptions"):
        for item in feed.get(key, []) or []:
            t = (item.get("text") or "").strip()
            if t:
                out.append(t)
    return out


def _images_from_feed(feed: dict) -> list[str]:
    out = []
    for img in feed.get("images", []) or []:
        url = img.get("url")
        if url:
            out.append(url)
    for vid in feed.get("videos", []) or []:
        thumb = vid.get("thumbnail_url")   # poster frame; keyframes added later
        if thumb:
            out.append(thumb)
    return out


def _video_ids_from_feed(feed: dict) -> list[str]:
    out = []
    for vid in feed.get("videos", []) or []:
        v = vid.get("video_id")
        if v:
            out.append(str(v))
    return out


def _from_story_spec(spec: dict) -> tuple[list[str], list[str]]:
    texts, images = [], []
    for block in ("link_data", "video_data", "photo_data", "template_data"):
        d = spec.get(block) or {}
        for tkey in ("message", "name", "title", "description", "link_description", "caption"):
            v = (d.get(tkey) or "").strip()
            if v:
                texts.append(v)
        for ikey in ("picture", "image_url"):
            v = d.get(ikey)
            if v:
                images.append(v)
    return texts, images


def _parse_creative(cr: dict, account_id: str, ad_name: str = "") -> Creative:
    feed = cr.get("asset_feed_spec") or {}
    story = cr.get("object_story_spec") or {}

    texts: list[str] = []
    images: list[str] = []
    video_ids: list[str] = []

    # flat fields (classic single-asset creatives)
    for v in (cr.get("title"), cr.get("body")):
        if v and v.strip():
            texts.append(v.strip())
    for v in (cr.get("image_url"), cr.get("thumbnail_url")):
        if v:
            images.append(v)
    if cr.get("video_id"):
        video_ids.append(str(cr["video_id"]))

    # dynamic (Advantage+) — the common Cuemath case
    texts += _texts_from_feed(feed)
    images += _images_from_feed(feed)
    video_ids += _video_ids_from_feed(feed)

    # classic page-post creatives
    s_texts, s_images = _from_story_spec(story)
    texts += s_texts
    images += s_images

    # dedup, preserve order
    texts = list(dict.fromkeys(texts))
    images = list(dict.fromkeys(images))
    video_ids = list(dict.fromkeys(video_ids))

    return Creative(
        creative_id=str(cr.get("id")),
        name=ad_name or cr.get("name") or "",
        ad_account_id=account_id,
        title=(texts[0] if texts else "")[:120],
        body=(texts[1] if len(texts) > 1 else (texts[0] if texts else ""))[:300],
        texts=texts,
        image_sources=images,
        video_ids=video_ids,
        is_active=True,
    )


# ------------------------------------------------------- video keyframes ------
def _sample_evenly(items: list, k: int) -> list:
    if len(items) <= k:
        return items
    step = len(items) / k
    return [items[int(i * step)] for i in range(k)]


def _select_frames(data: list[dict], k: int) -> list[str]:
    uris = [t["uri"] for t in data if t.get("uri")]
    preferred = [t["uri"] for t in data if t.get("is_preferred") and t.get("uri")]
    rest = [u for u in uris if u not in preferred]
    return (preferred + _sample_evenly(rest, max(k - len(preferred), 0)))[:k]


def fetch_video_frames(ad_account_id: str, video_ids: list[str],
                       k: Optional[int] = None) -> dict[str, list[str]]:
    """Resolve video_id → up to k keyframe URLs via the account-scoped `advideos`
    edge (the video-node `thumbnails` edge needs a scope `ads_read` tokens lack).
    Pages newest-first, stopping once every needed id is found. Robust to failures."""
    k = k or config.MAX_FRAMES_PER_VIDEO
    needed = {v for v in video_ids if v}
    result: dict[str, list[str]] = {}
    url = f"{config.GRAPH_BASE}/act_{ad_account_id}/advideos"
    params = {"fields": "id,thumbnails{uri,is_preferred}", "limit": 50,
              "access_token": config.META_ACCESS_TOKEN}
    pages = 0
    while url and needed and pages < config.VIDEO_PAGE_BUDGET:
        try:
            resp = requests.get(url, params=params, timeout=45)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:  # noqa: BLE001 - never log the URL (it carries the token)
            print(f"[meta] advideos fetch failed: {type(exc).__name__} — falling back to poster frames")
            break
        for v in payload.get("data", []):
            vid = str(v.get("id"))
            if vid in needed:
                frames = _select_frames((v.get("thumbnails") or {}).get("data", []) or [], k)
                if frames:
                    result[vid] = frames
                needed.discard(vid)
        pages += 1
        # Active videos, if advideos-listed at all, are among the newest. If page 1
        # matched none, these are dynamic placement-asset videos not in advideos —
        # stop rather than page fruitlessly (falls back to poster frames).
        if not result:
            break
        url = payload.get("paging", {}).get("next")
        params = {}  # `next` is fully-formed
    return result


# ------------------------------------------------------------------- fetch ----
def fetch_active_creatives(ad_account_id: str, limit: int = 200) -> list[Creative]:
    """Return ACTIVE ads' creatives (parsed) for an account. Mock when no token."""
    if not config.META_ACCESS_TOKEN:
        return _mock_creatives(ad_account_id)

    url = f"{config.GRAPH_BASE}/act_{ad_account_id}/ads"
    params = {
        "fields": f"name,effective_status,creative{{{_CREATIVE_FIELDS}}}",
        "effective_status": '["ACTIVE"]',
        "limit": min(limit, 100),
        "access_token": config.META_ACCESS_TOKEN,
    }
    out: list[Creative] = []
    seen: set[str] = set()
    while url:
        resp = requests.get(url, params=params, timeout=45)
        _raise_for_meta(resp)
        payload = resp.json()
        for ad in payload.get("data", []):
            cr = ad.get("creative") or {}
            cid = cr.get("id")
            if not cid or cid in seen:
                continue
            seen.add(cid)
            parsed = _parse_creative(cr, ad_account_id, ad_name=ad.get("name", ""))
            if parsed.texts or parsed.image_sources or parsed.video_ids:   # skip empties
                out.append(parsed)
            if len(out) >= limit:
                break
        if len(out) >= limit:
            break
        url = payload.get("paging", {}).get("next")
        params = {}  # `next` is a fully-formed URL

    _enrich_video_frames(out)
    return out


def _enrich_video_frames(creatives: list[Creative]) -> None:
    """Add multi-frame video keyframes to each creative's image_sources (in place)."""
    all_vids = [v for c in creatives for v in c.video_ids]
    if not all_vids:
        return
    account_id = creatives[0].ad_account_id
    frames = fetch_video_frames(account_id, all_vids)
    for c in creatives:
        extra = [u for v in c.video_ids for u in frames.get(v, [])]
        if extra:
            # keyframes first (richer signal), then the poster thumbs already present
            c.image_sources = list(dict.fromkeys(extra + c.image_sources))


# -------------------------------------------------------------------- mock ----
def _mock_creatives(ad_account_id: str) -> list[Creative]:
    samples = [
        ("m1", "Live 1:1 Math Tutoring",
         "Book a FREE trial class today! Personalised 1-on-1 online math tutoring for grades 1-10.",
         "https://picsum.photos/seed/cuemath1/512/512"),
        ("m2", "Free Math Trial Class",
         "Start with a FREE trial! One-on-one online maths classes for your child, grades 1 to 10.",
         "https://picsum.photos/seed/cuemath1/512/512"),  # near-dup of m1
        ("m3", "Build Math Confidence",
         "Help your child fall in love with numbers. Certified tutors, live classes, real results.",
         "https://picsum.photos/seed/cuemath2/512/512"),
        ("m4", "Coding for Kids",
         "Beyond math — now teaching coding & logic for young learners. Try a free session.",
         "https://picsum.photos/seed/cuemath3/512/512"),
    ]
    return [
        Creative(
            creative_id=cid, name=name, ad_account_id=ad_account_id,
            title=name, body=body, texts=[name, body], image_sources=[img], is_active=True,
        )
        for cid, name, body, img in samples
    ]


def using_mock() -> bool:
    return not config.META_ACCESS_TOKEN
