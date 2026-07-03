"""Creative Similarity Score — Streamlit prototype.

Two modes off one embedding index:
  1. Pre-flight  — score a NEW/candidate creative before launch (input form).
  2. Monitoring  — all-pairs scan of the ACTIVE set to flag existing cannibalization.

Run:  streamlit run app.py
Docs: 04-Technical/2026-07-03-meta-creative-similarity-score-replication.md
"""
from __future__ import annotations

import time

import pandas as pd
import streamlit as st

import config
import store
from meta_client import fetch_active_creatives, using_mock
from similarity import ensure_embeddings, score_candidate
from embeddings import embed_image, embed_text

st.set_page_config(page_title="Creative Similarity Score", page_icon="🎯", layout="wide")

VERDICT_COLOR = {"unique": "🟢", "moderate": "🟡", "high-overlap": "🔴"}


# --- active-set index (cached per account) -------------------------------------
def get_active_index(account_id: str, force: bool = False):
    if not force:
        cached = store.load_embeddings(account_id)
        if cached:
            return cached
    creatives = fetch_active_creatives(account_id)
    with st.spinner(f"Embedding {len(creatives)} active creatives (visual + copy)…"):
        ensure_embeddings(creatives)
    store.save_embeddings(account_id, creatives)
    return creatives


# --- sidebar -------------------------------------------------------------------
st.sidebar.title("🎯 Creative Similarity")
account_id = st.sidebar.selectbox(
    "Ad account",
    options=list(config.AD_ACCOUNTS.keys()),
    format_func=lambda a: f"{config.AD_ACCOUNTS[a]} ({a})",
    index=list(config.AD_ACCOUNTS).index(config.DEFAULT_ACCOUNT),
)
w_visual = st.sidebar.slider("Visual weight", 0.0, 1.0, config.DEFAULT_VISUAL_WEIGHT, 0.05)
w_text = round(1.0 - w_visual, 2)
st.sidebar.caption(f"Copy weight = **{w_text}** (weights sum to 1)")
top_n = st.sidebar.slider("Neighbours to show", 1, 10, config.TOP_N)

if st.sidebar.button("🔄 Rebuild active index"):
    get_active_index(account_id, force=True)
    st.sidebar.success("Rebuilt.")

if using_mock():
    st.sidebar.warning("No META_ACCESS_TOKEN — running on **mock** creatives. "
                       "Set the token in .env for live data.")
else:
    st.sidebar.info("Live Meta Marketing API mode.")

active = get_active_index(account_id)
st.sidebar.metric("Active creatives indexed", len(active))

# --- header --------------------------------------------------------------------
st.title("Creative Similarity Score")
st.caption("Our replication of Meta's Andromeda / Entity-ID similarity — explainable, "
           "per-modality (visual vs copy). High score ⇒ likely cannibalization.")

tab_preflight, tab_monitor, tab_history = st.tabs(
    ["🚀 Pre-flight (score a new creative)", "📡 Monitoring (active set)", "🗂 Score history"]
)


def render_neighbors(title: str, neighbors):
    st.markdown(f"**{title}**")
    if not neighbors:
        st.caption("— none (modality missing on candidate) —")
        return
    for n in neighbors:
        cols = st.columns([1, 4])
        with cols[0]:
            if n.image_url:
                st.image(n.image_url, width=90)
        with cols[1]:
            st.markdown(f"`{n.similarity*100:5.1f}%` · **{n.name}** ({n.creative_id})")
            if n.title or n.body:
                st.caption((n.title + " — " + n.body)[:160])


# ================================ PRE-FLIGHT ==================================
with tab_preflight:
    st.subheader("Score a candidate creative against the active set")
    with st.form("preflight"):
        c1, c2 = st.columns(2)
        with c1:
            headline = st.text_input("Headline (title)")
            primary_text = st.text_area("Primary text (body)", height=120)
        with c2:
            up = st.file_uploader("Creative image", type=["png", "jpg", "jpeg", "webp"])
            img_url = st.text_input("…or image URL")
        submitted = st.form_submit_button("Score it")

    if submitted:
        img_bytes = up.read() if up is not None else (img_url or None)
        cand_visual = embed_image(img_bytes) if img_bytes else None
        cand_copy = embed_text(headline, primary_text)

        if cand_visual is None and cand_copy is None:
            st.error("Give me at least an image or some copy.")
        else:
            res = score_candidate(cand_visual, cand_copy, active,
                                  w_visual=w_visual, w_text=w_text, top_n=top_n)
            store.append_score(account_id, res, submission_id=f"ui-{int(time.time())}")

            m1, m2, m3 = st.columns(3)
            m1.metric("Similarity Score", f"{res.similarity_score}/100",
                      help="Composite: weighted max visual & copy similarity")
            m2.metric("Verdict", f"{VERDICT_COLOR.get(res.verdict,'')} {res.verdict}")
            m3.metric("Visual / Copy max",
                      f"{(res.visual_sim_max or 0)*100:.0f}% / {(res.copy_sim_max or 0)*100:.0f}%")

            if res.verdict == "high-overlap" and res.nearest_creative_id:
                st.warning(f"Closest match is **{res.nearest_creative_id}** "
                           f"({res.nearest_modality}) — Meta would likely group these "
                           f"under one Entity ID and let them cannibalize each other.")

            st.divider()
            g1, g2 = st.columns(2)
            with g1:
                render_neighbors("👁 Visual similarity", res.visual_neighbors)
            with g2:
                render_neighbors("✍️ Copy similarity", res.copy_neighbors)


# ================================ MONITORING ==================================
with tab_monitor:
    st.subheader("Cannibalization scan across the active set")
    st.caption("Each active creative scored vs every OTHER active creative (self excluded).")
    if st.button("Run scan"):
        rows = []
        prog = st.progress(0.0)
        for i, c in enumerate(active):
            res = score_candidate(c.visual_vec, c.copy_vec, active,
                                  exclude_id=c.creative_id,
                                  w_visual=w_visual, w_text=w_text, top_n=1)
            store.append_score(account_id, res, creative_id=c.creative_id, is_active=True)
            rows.append({
                "creative_id": c.creative_id, "name": c.name,
                "score": res.similarity_score, "verdict": res.verdict,
                "visual_max": round((res.visual_sim_max or 0) * 100, 1),
                "copy_max": round((res.copy_sim_max or 0) * 100, 1),
                "nearest": res.nearest_creative_id, "nearest_modality": res.nearest_modality,
            })
            prog.progress((i + 1) / max(len(active), 1))
        df = pd.DataFrame(rows).sort_values("score", ascending=False)
        st.dataframe(df, width="stretch", hide_index=True)
        n_high = int((df["verdict"] == "high-overlap").sum())
        st.metric("High-overlap creatives", n_high)


# ================================ HISTORY =====================================
with tab_history:
    st.subheader("creative_similarity table (local parquet → prod Redshift)")
    df = store.read_scores()
    if df.empty:
        st.caption("No scores yet — run a pre-flight or a scan.")
    else:
        st.dataframe(df.sort_values("scored_at", ascending=False),
                     width="stretch", hide_index=True)
        st.caption(f"{len(df)} rows · {config.SCORES_TABLE}")
