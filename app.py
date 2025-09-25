import os
import io
import json
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Configuration
# -----------------------------
DATA_PATH = os.environ.get("CMA_DATA_PATH", "data/artworks.csv")
DEFAULT_TOPK = 10
HYBRID_DEFAULT_WEIGHT = 0.6  # weight for semantic SBERT; TF-IDF weight = 1 - this
RANDOM_SEED = 42

st.set_page_config(page_title="CMA — AI Analyzer (Robust)", page_icon="🎨", layout="wide")
np.random.seed(RANDOM_SEED)

@st.cache_data
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    # Normalize text fields (fillna, strip)
    for c in ["title", "artist", "medium", "department", "description", "tags"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    # Coerce year to int where possible
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
    # Ensure id exists
    if "id" not in df.columns:
        df["id"] = np.arange(1, len(df) + 1)
    return df

df = load_data()

########## Embedding & Similarity ##########
def _try_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model, None
    except Exception as e:
        return None, str(e)

@st.cache_resource
def get_embedder():
    model, err = _try_sentence_transformers()
    if model is not None:
        return ("sbert", model, None)
    else:
        return ("tfidf", None, err or "SentenceTransformers unavailable")

embedder_kind, embedder_model, embedder_error = get_embedder()

@st.cache_data(show_spinner=False)
def build_index_text_corpus(_df):
    cols = ["title","artist","year","medium","department","description","tags"]
    present = [c for c in cols if c in _df.columns]
    combined = _df[present].astype(str).agg(" | ".join, axis=1)
    return combined.tolist()

corpus = build_index_text_corpus(df)

@st.cache_resource(show_spinner=False)
def build_sbert_embeddings(_corpus):
    embs = np.array(embedder_model.encode(_corpus, show_progress_bar=False))
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
    return embs / norms

@st.cache_resource(show_spinner=False)
def build_tfidf_matrix(_corpus):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(ngram_range=(1,2), max_features=8000, min_df=1)
    X = vec.fit_transform(_corpus)
    return vec, X

sbert_embeddings = None
if embedder_kind == "sbert":
    try:
        sbert_embeddings = build_sbert_embeddings(corpus)
    except Exception as e:
        sbert_embeddings = None
        embedder_error = f"SBERT indexing failed: {e}"
        embedder_kind = "tfidf"

tfidf_vec, tfidf_X = build_tfidf_matrix(corpus)

def encode_query_sbert(q):
    v = np.array(embedder_model.encode([q], show_progress_bar=False))[0]
    v = v / (np.linalg.norm(v) + 1e-12)
    return v

def encode_query_tfidf(q):
    return tfidf_vec.transform([q])

def norm01(x):
    if x is None: return None
    x = np.asarray(x, dtype=float)
    if x.size == 0: return x
    lo = np.percentile(x, 5)
    hi = np.percentile(x, 95)
    if np.isclose(lo, hi):
        return np.zeros_like(x)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0)


def safe_topk_union(k, sbert_scores=None, tfidf_scores=None, weight=HYBRID_DEFAULT_WEIGHT):
    if sbert_scores is not None:
        sbert_scores = norm01(sbert_scores)
    if tfidf_scores is not None:
        tfidf_scores = norm01(tfidf_scores)
    if sbert_scores is not None and tfidf_scores is not None:
        hybrid = weight * sbert_scores + (1.0 - weight) * tfidf_scores
        order = np.argsort(-hybrid)[:k]
        return order, hybrid[order]
    elif sbert_scores is not None:
        order = np.argsort(-sbert_scores)[:k]
        return order, sbert_scores[order]
    else:
        order = np.argsort(-tfidf_scores)[:k]
        return order, tfidf_scores[order]

########## Sidebar Controls ##########
with st.sidebar:
    st.header("🔎 Search & Filters")
    query = st.text_input("Semantic search", placeholder="e.g., sculpture of an animal in bronze").strip()
    yr_min, yr_max = int(df["year"].min()), int(df["year"].max())
    if yr_min > yr_max:
        yr_min, yr_max = 0, 0
    year_range = st.slider("Year range", yr_min, yr_max, (yr_min, yr_max))

    def _opts(series):
        vals = sorted({str(x).strip() for x in series if str(x).strip()})
        return ["(any)"] + vals

    artist_sel = st.selectbox("Artist", _opts(df["artist"]))
    medium_sel = st.selectbox("Medium", _opts(df["medium"]))
    dept_sel = st.selectbox("Department", _opts(df["department"]))

    k = st.number_input("Top K results", min_value=1, max_value=50, value=DEFAULT_TOPK, step=1)
    use_hybrid = st.checkbox("Hybrid ranking (Semantic + TF-IDF)", value=True)
    hybrid_w = st.slider("Hybrid semantic weight", 0.0, 1.0, HYBRID_DEFAULT_WEIGHT, 0.05, disabled=not use_hybrid)

    st.markdown("---")
    st.caption(f"Embedder mode: **{'SentenceTransformers' if (embedder_kind=='sbert' and sbert_embeddings is not None) else 'TF-IDF only'}**")
    if embedder_error and (embedder_kind != 'sbert' or sbert_embeddings is None):
        with st.expander("Why semantic mode is disabled"):
            st.code(str(embedder_error))
    st.caption(f"Records indexed: **{len(df)}**")

########## Filtering ##########
def apply_filters(_df):
    mask = (_df["year"] >= year_range[0]) & (_df["year"] <= year_range[1])
    if artist_sel != "(any)":
        mask &= (_df["artist"] == artist_sel)
    if medium_sel != "(any)":
        mask &= (_df["medium"] == medium_sel)
    if dept_sel != "(any)":
        mask &= (_df["department"] == dept_sel)
    return _df[mask]

filtered_df = apply_filters(df)
filtered_idx = filtered_df.index.to_numpy()

########## Search ##########
st.title("🎨 Cleveland Museum of Art — AI-Assisted Analyzer (Robust)")
st.write("Hybrid semantic + keyword retrieval, curator notes, CSV export, and basic analytics with graceful fallbacks.")

results_df = filtered_df.copy()
scores = None

def fullscore_sbert(q):
    qv = encode_query_sbert(q)
    return sbert_embeddings @ qv

def fullscore_tfidf(q):
    qv = encode_query_tfidf(q)
    return (tfidf_X @ qv.T).toarray().ravel()

if query:
    try:
        sbert_full = fullscore_sbert(query) if (sbert_embeddings is not None) else None
    except Exception:
        sbert_full = None
    tfidf_full = fullscore_tfidf(query)

    sbert_sub = sbert_full[filtered_idx] if sbert_full is not None else None
    tfidf_sub = tfidf_full[filtered_idx] if tfidf_full is not None else None

    if use_hybrid and (sbert_sub is not None):
        order_local, sc_local = safe_topk_union(int(k), sbert_sub, tfidf_sub, hybrid_w)
    elif sbert_sub is not None:
        order_local = np.argsort(-sbert_sub)[:int(k)]
        sc_local = sbert_sub[order_local]
    else:
        order_local = np.argsort(-tfidf_sub)[:int(k)]
        sc_local = tfidf_sub[order_local]

    chosen_indices = filtered_idx[order_local] if len(filtered_idx) > 0 else np.array([], dtype=int)
    results_df = df.loc[chosen_indices].copy()
    scores = sc_local.tolist()

########## Results Table ##########
def render_row(row, score=None):
    left, right = st.columns([2, 1])
    with left:
        st.subheader(f"{row.title} ({int(row.year)})")
        st.write(f"**Artist:** {row.artist}  |  **Medium:** {row.medium}  |  **Department:** {row.department}")
        st.write(row.description)
        if isinstance(row.tags, str) and row.tags.strip():
            st.caption("Tags: " + ", ".join(t.strip() for t in str(row.tags).split(";") if t.strip()))
    with right:
        if score is not None and np.isfinite(score):
            st.metric("Score", f"{float(score):.3f}")
        st.caption(f"ID: {row.id}")

if results_df.empty:
    st.info("No results. Try broadening filters or adjusting the query.")
else:
    st.write("### Results")
    for i, (_, r) in enumerate(results_df.iterrows()):
        sc = scores[i] if (scores is not None and i < len(scores)) else None
        render_row(r, sc)
        st.divider()

########## Curator Notes (by TITLE) ##########
st.write("### Curator Notes")
if "notes" not in st.session_state:
    st.session_state["notes"] = {}  # title -> note text

title_options = results_df["title"].tolist() if not results_df.empty else df["title"].tolist()
selected_title = st.selectbox("Select artwork by title", title_options, key="note_title_select")

# Pre-fill with any existing note for this title
existing = st.session_state["notes"].get(selected_title, "")
note_text = st.text_area("Notes", value=existing, key="note_text_area", placeholder="Ex: Consider for Spring exhibition on maritime themes.")

cols_btn = st.columns([1,1,2,2])
with cols_btn[0]:
    if st.button("Save note", key="save_note_btn"):
        st.session_state["notes"][selected_title] = note_text
        st.success(f"Saved note for “{selected_title}”.")

with cols_btn[1]:
    if st.button("Clear note", key="clear_note_btn"):
        st.session_state["notes"].pop(selected_title, None)
        st.info(f"Cleared note for “{selected_title}”.")

# Download single note as TXT
def _slugify(s):
    return "".join(c if c.isalnum() or c in ("-","_") else "_" for c in s)[:80]

single_txt = ""
if selected_title in st.session_state["notes"]:
    single_txt = f"Title: {selected_title}\n\nNote:\n{st.session_state['notes'][selected_title]}\n"
st.download_button(
    "Download this note (.txt)",
    data=single_txt.encode("utf-8"),
    file_name=f"{_slugify(selected_title)}_note.txt",
    mime="text/plain",
    disabled=(selected_title not in st.session_state["notes"]),
    key="dl_single_note"
)

# Download all notes as TXT
if st.session_state["notes"]:
    all_txt_parts = []
    for t, n in st.session_state["notes"].items():
        all_txt_parts.append(f"Title: {t}\nNote:\n{n}\n" + "-"*40 + "\n")
    all_txt = "".join(all_txt_parts)
else:
    all_txt = "No notes yet.\n"
st.download_button(
    "Download ALL notes (.txt)",
    data=all_txt.encode("utf-8"),
    file_name="curator_notes_all.txt",
    mime="text/plain",
    key="dl_all_notes"
)

########## Export (CSV) ##########
st.write("### Export")
export_cols = ["id", "title", "artist", "year", "medium", "department", "description", "tags"]
df_with_notes = df.copy()
df_with_notes["notes"] = df_with_notes["title"].map(lambda t: st.session_state["notes"].get(t, ""))

export_choice = st.selectbox("What to export?", ["Current results", "All (with notes column)"])
if export_choice == "Current results":
    out_df = results_df[export_cols].copy() if not results_df.empty else df[export_cols].head(0).copy()
    out_df["notes"] = out_df["title"].map(lambda t: st.session_state["notes"].get(t, ""))
else:
    out_df = df_with_notes[export_cols + ["notes"]].copy()

csv_buf = io.StringIO()
out_df.to_csv(csv_buf, index=False, encoding="utf-8")
st.download_button("Download CSV", csv_buf.getvalue(), file_name="museum_export.csv", mime="text/csv")

########## Quick Analytics ##########
st.write("### Quick Analytics")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total works", len(df))
with col2:
    st.metric("Artists (unique)", df["artist"].nunique())
with col3:
    st.metric("Departments", df["department"].nunique())

dept_counts = df["department"].value_counts().reset_index()
dept_counts.columns = ["department", "count"]
st.bar_chart(dept_counts.set_index("department"))

########## Diagnostics ##########
with st.expander("⚙️ Diagnostics"):
    st.write({
        "embedder_kind": embedder_kind,
        "sbert_available": sbert_embeddings is not None,
        "records": len(df),
        "hybrid_enabled": use_hybrid if 'use_hybrid' in locals() else None,
        "hybrid_weight": hybrid_w if 'hybrid_w' in locals() else None,
        "notes_count": len(st.session_state.get('notes', {})),
    })

