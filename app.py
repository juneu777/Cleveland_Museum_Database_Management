import os
import io
import json
import numpy as np
import pandas as pd
import streamlit as st

DATA_PATH = os.environ.get("CMA_DATA_PATH", "data/data.csv")
DEFAULT_TOPK = 10
HYBRID_DEFAULT_WEIGHT = 0.6  
RANDOM_SEED = 42
score_label = "Relevance (%)"




st.set_page_config(page_title="CMA — AI Analyzer (Robust)", page_icon="🎨", layout="wide")
np.random.seed(RANDOM_SEED)

@st.cache_data
def load_data(path=DATA_PATH, max_rows=None):
    USECOLS = [
        "id", "accession_number", "title", "creators", "creation_date",
        "creation_date_earliest", "technique", "department",
        "wall_description", "description", "tombstone",
        "artists_tags", "culture", "type", "collection",
        "share_license_status", "image_web", "current_location", "url", "creditline"
    ]
    # Only keep existing ones
    probe = pd.read_csv(path, nrows=0, low_memory=False)
    usecols = [c for c in USECOLS if c in probe.columns]

    df = pd.read_csv(
        path,
        usecols=usecols,
        nrows=max_rows,               
        low_memory=False,             
        engine="pyarrow" if "pyarrow" in pd.__dict__.get("options", {}).__class__.__module__ else None, 
    )

    def _artist_from_creators(val):
        if isinstance(val, list):
            return ", ".join([str(getattr(x, "get", lambda k, d=None: None)("description", "") or x).strip() for x in val if str(x).strip()])
        if isinstance(val, str):
            v = val.strip()
            if (v.startswith("[") and v.endswith("]")) or (v.startswith("{") and v.endswith("}")):
                try:
                    import ast
                    parsed = ast.literal_eval(v)
                    if isinstance(parsed, list):
                        return ", ".join([str(getattr(x, "get", lambda k, d=None: None)("description", "") or x).strip() for x in parsed if str(x).strip()])
                except Exception:
                    pass
            return v
        return ""

    def _year_from_row(r):
        import re
        cd = str(r.get("creation_date", "") or "")
        m = re.search(r"(\d{3,4})", cd)
        if m: return int(m.group(1))
        ce = r.get("creation_date_earliest")
        try:
            return int(ce)
        except Exception:
            return 0

    def _desc_from_row(r):
        return (r.get("wall_description") or r.get("description") or r.get("tombstone") or "").strip()

    def _tags_from_row(r):
        fields = [r.get("artists_tags"), r.get("culture"), r.get("type"),
                  r.get("collection"), r.get("technique"), r.get("department"),
                  r.get("share_license_status")]
        out = []
        for f in fields:
            if isinstance(f, list): out.extend([str(x).strip() for x in f if str(x).strip()])
            elif isinstance(f, str): out.append(f.strip())
        seen = set(); tags = []
        for t in out:
            if t and t not in seen:
                seen.add(t); tags.append(t)
        return ";".join(tags)

    if "artist" not in df.columns:
        df["artist"] = df.get("creators", "").apply(_artist_from_creators)
    if "year" not in df.columns:
        df["year"] = df.apply(_year_from_row, axis=1)
    if "medium" not in df.columns:
        df["medium"] = df.get("technique", "").fillna("").astype(str)
    if "description" not in df.columns:
        df["description"] = df.apply(_desc_from_row, axis=1)
    if "tags" not in df.columns:
        df["tags"] = df.apply(_tags_from_row, axis=1)
    if "image_web" not in df.columns:
        df["image_web"] = ""

    for c in ["title", "artist", "medium", "department", "description", "tags", "image_web"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)

    if "id" not in df.columns:
        if "accession_number" in df.columns:
            df["id"] = df["accession_number"]
        else:
            df["id"] = np.arange(1, len(df) + 1)

    return df


MAX_ROWS = int(os.environ.get("CMA_MAX_ROWS", "0")) or None
df = load_data(DATA_PATH, max_rows=MAX_ROWS)
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
    combined = (
        _df["title"].fillna("") + " " +
        _df["title"].fillna("") + " " +  
        _df["description"].fillna("") + " " +
        _df["tags"].fillna("") + " " +
        _df["medium"].fillna("")
    )
    return combined.tolist()

corpus = build_index_text_corpus(df)

@st.cache_resource(show_spinner=False)
def build_sbert_embeddings(_corpus):
    embs = np.array(
        embedder_model.encode(_corpus, show_progress_bar=False),
        dtype=np.float32
    )

    embs = np.nan_to_num(embs, nan=0.0, posinf=0.0, neginf=0.0)

    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.where((~np.isfinite(norms)) | (norms < 1e-12), 1.0, norms)

    embs = embs / norms

    if not np.isfinite(embs).all():
        raise ValueError("sbert_embeddings still contains NaN or inf after normalization")

    return embs

@st.cache_resource(show_spinner=False)
def build_tfidf_matrix(_corpus):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=8000,
    min_df=1,
    norm="l2",          
    sublinear_tf=True   
)
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
    q = str(q).strip()
    if not q:
        return np.zeros(sbert_embeddings.shape[1], dtype=np.float32)

    v = embedder_model.encode(
        [q],
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False
    )[0].astype(np.float32)

    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

    norm = np.linalg.norm(v)
    if not np.isfinite(norm) or norm < 1e-12:
        return np.zeros_like(v)

    v = v / norm
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

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
    """
    Inputs must already be 0..1 (we normalize globally before filtering).
    Blend using `weight` and return top-k order and blended scores (0..1).
    """
    if (sbert_scores is not None) and (tfidf_scores is not None):
        hybrid = weight * sbert_scores + (1.0 - weight) * tfidf_scores
        order = np.argsort(-hybrid)[:k]
        return order, hybrid[order]
    elif sbert_scores is not None:
        order = np.argsort(-sbert_scores)[:k]
        return order, sbert_scores[order]
    else:
        order = np.argsort(-tfidf_scores)[:k]
        return order, tfidf_scores[order]


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

    st.markdown("---")
    cc0_only = st.checkbox("Open Access (CC0) only", value=False, disabled=("share_license_status" not in df.columns))
    has_image_only = st.checkbox("Has image only", value=False, disabled=("image_web" not in df.columns))
    on_view_only = st.checkbox("On view (in galleries) only", value=False, disabled=("current_location" not in df.columns))

def apply_filters(_df):
    mask = (_df["year"] >= year_range[0]) & (_df["year"] <= year_range[1])
    if artist_sel != "(any)":
        mask &= (_df["artist"] == artist_sel)
    if medium_sel != "(any)":
        mask &= (_df["medium"] == medium_sel)
    if dept_sel != "(any)":
        mask &= (_df["department"] == dept_sel)

    if "share_license_status" in _df.columns and cc0_only:
        mask &= (_df["share_license_status"].astype(str).str.upper() == "CC0")
    if "image_web" in _df.columns and has_image_only:
        mask &= _df["image_web"].astype(str).str.strip().ne("")
    if "current_location" in _df.columns and on_view_only:
        mask &= _df["current_location"].astype(str).str.strip().ne("")

    return _df[mask]


filtered_df = apply_filters(df)
filtered_idx = filtered_df.index.to_numpy()

def rerank_and_dedup_results(results_df, scores):
    if results_df.empty:
        return results_df, scores

    tmp = results_df.copy()
    tmp["_score"] = list(scores) if scores is not None else [0.0] * len(tmp)

    generic_titles = {
        "untitled",
        "no title",
        "study",
        "fragment"
    }

    title_lower = tmp["title"].fillna("").str.strip().str.lower()
    tmp.loc[title_lower.isin(generic_titles), "_score"] -= 15.0

    tmp["_dedup_key"] = (
        tmp["title"].fillna("").str.strip().str.lower() + " | " +
        tmp["artist"].fillna("").str.strip().str.lower()
    )

    tmp = tmp.sort_values("_score", ascending=False)
    tmp = tmp.drop_duplicates(subset="_dedup_key", keep="first")

    tmp = tmp.sort_values("_score", ascending=False).reset_index(drop=True)

    new_scores = tmp["_score"].tolist()
    tmp = tmp.drop(columns=["_score", "_dedup_key"])

    return tmp, new_scores


def apply_query_term_boost(results_df, scores, query):
    if results_df.empty or not query.strip():
        return results_df, scores

    tmp = results_df.copy()
    tmp["_score"] = list(scores) if scores is not None else [0.0] * len(tmp)

    q_terms = query.lower().split()

    searchable = (
        tmp["title"].fillna("").str.lower() + " " +
        tmp["description"].fillna("").str.lower() + " " +
        tmp["tags"].fillna("").str.lower() + " " +
        tmp["medium"].fillna("").str.lower()
    )

    for term in q_terms:
        tmp.loc[searchable.str.contains(term, na=False), "_score"] += 0.05

    tmp = tmp.sort_values("_score", ascending=False).reset_index(drop=True)
    new_scores = tmp["_score"].tolist()
    tmp = tmp.drop(columns=["_score"])

    return tmp, new_scores

st.title("Semantic Search Engine for the Cleveland Museum of Art Collection")
st.write("Hybrid semantic + keyword retrieval, curator notes, CSV export, and basic analytics with graceful fallbacks.")

results_df = filtered_df.copy()
scores = None

def fullscore_sbert(q):
    qv = encode_query_sbert(q)

    if not np.isfinite(qv).all():
        print("BAD QUERY VECTOR:", qv)
        return np.zeros(len(sbert_embeddings), dtype=np.float32)

    scores = sbert_embeddings @ qv
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    if not np.isfinite(scores).all():
        print("BAD SCORES AFTER MATMUL")
        return np.zeros(len(sbert_embeddings), dtype=np.float32)

    return scores



def fullscore_tfidf(q):
    qv = encode_query_tfidf(q)
    return (tfidf_X @ qv.T).toarray().ravel()

if query:
    try:
        sbert_full = fullscore_sbert(query) if (sbert_embeddings is not None) else None
    except Exception:
        sbert_full = None
    tfidf_full = fullscore_tfidf(query)
    s_full_01 = norm01((sbert_full + 1.0) / 2.0) if sbert_full is not None else None
    t_full_01 = norm01(tfidf_full)

    sbert_sub = s_full_01[filtered_idx] if s_full_01 is not None else None
    tfidf_sub = t_full_01[filtered_idx] if t_full_01 is not None else None

    candidate_k = min(len(filtered_idx), max(int(k) * 5, 50))

    if use_hybrid and (sbert_sub is not None):
        order_local, sc_local_01 = safe_topk_union(candidate_k, sbert_sub, tfidf_sub, hybrid_w)
        sc_local = sc_local_01 
        score_label = f"Relevance (Hybrid, {int(hybrid_w*100)}% semantic)"
    elif sbert_sub is not None:
        order_local = np.argsort(-sbert_sub)[:candidate_k]
        sc_local = sbert_sub[order_local]
        score_label = "Relevance (SBERT %)"
    else:
        order_local = np.argsort(-tfidf_sub)[:candidate_k]
        sc_local = tfidf_sub[order_local] 
        score_label = "Relevance (TF-IDF %)"

    chosen_indices = filtered_idx[order_local] if len(filtered_idx) > 0 else np.array([], dtype=int)
    results_df = df.loc[chosen_indices].copy()
    scores = sc_local.tolist()

    results_df["_score"] = scores

    stop_terms = {
    "art", "artwork", "work", "piece", "image",
    "drawing", "drawings", "painting", "paintings"
    }

    q_terms = [t for t in query.lower().split() if t not in stop_terms]

    if not q_terms:
        q_terms = query.lower().split()

    title_text = results_df["title"].fillna("").str.lower()
    tags_text = results_df["tags"].fillna("").str.lower()
    desc_text = results_df["description"].fillna("").str.lower()
    medium_text = results_df["medium"].fillna("").str.lower()

    for term in q_terms:
        results_df.loc[title_text.str.contains(term, na=False, regex=False), "_score"] += 0.12
        results_df.loc[tags_text.str.contains(term, na=False, regex=False), "_score"] += 0.08
        results_df.loc[medium_text.str.contains(term, na=False, regex=False), "_score"] += 0.02
        results_df.loc[desc_text.str.contains(term, na=False, regex=False), "_score"] += 0.02

    results_df["_score"] = results_df["_score"].clip(0.0, 1.0)

    primary_terms = [t for t in query.lower().split() if t not in {"drawing", "drawings", "painting", "paintings", "art", "artwork"}]

    if primary_terms:
        subject_text = (
            results_df["title"].fillna("").str.lower() + " " +
            results_df["tags"].fillna("").str.lower()
        )

        for term in primary_terms:
            results_df.loc[subject_text.str.contains(term, na=False), "_score"] += 0.15
    
    
    generic_titles = {"untitled", "no title", "study", "fragment"}
    title_lower = results_df["title"].fillna("").str.strip().str.lower()
    results_df.loc[title_lower.isin(generic_titles), "_score"] -= 0.15
    results_df["_score"] = results_df["_score"].clip(0.0, 1.0)

    results_df["_dedup_key"] = (
        results_df["title"].fillna("").str.strip().str.lower() + " | " +
        results_df["artist"].fillna("").str.strip().str.lower()
    )

    results_df = results_df.sort_values("_score", ascending=False)
    results_df = results_df.drop_duplicates(subset="_dedup_key", keep="first")

    results_df = results_df[
        ~results_df["title"].fillna("").str.strip().str.lower().str.startswith("untitled")
    ]

    results_df = results_df.sort_values("_score", ascending=False).head(int(k)).copy()

    scores = results_df["_score"].tolist()
    results_df = results_df.drop(columns=["_score", "_dedup_key"])

    q_norm = query.strip().lower()
    if results_df.shape[0] > 0:
        title_matches = results_df["title"].fillna("").str.strip().str.lower() == q_norm
        if title_matches.any():
            match_idx = np.where(title_matches.values)[0].tolist()
            non_idx = np.where(~title_matches.values)[0].tolist()

            results_df = pd.concat(
                [results_df.iloc[match_idx], results_df.iloc[non_idx]],
                ignore_index=True
            )

            scores = [100.0] * len(match_idx) + [scores[i] for i in non_idx]

def render_row(row, score=None):
    left, right = st.columns([2, 1])

    with left:
        st.subheader(f"{row.title} ({int(row.year)})")
        st.write(f"**Artist:** {row.artist}  |  **Medium:** {row.medium}  |  **Department:** {row.department}")

        st.write(row.description or "—")

        if "creditline" in row.index and str(row.creditline).strip():
            st.caption(f"Creditline: {row.creditline}")

        if isinstance(row.tags, str) and row.tags.strip():
            st.caption("Tags: " + ", ".join(t.strip() for t in str(row.tags).split(";") if t.strip()))

        if "url" in row.index and str(row.url).strip():
            st.markdown(f"[View on CMA]({row.url})")

    with right:
        img = str(getattr(row, "image_web", "") or "").strip()
        if img:
            st.image(img, use_container_width=True)

        if score is not None and np.isfinite(score):
            score_percent = float(score) * 100
            st.metric(score_label, f"{score_percent:.1f}%")

        st.caption(f"ID: {row.id}")


if results_df.empty:
    st.info("No results. Try broadening filters or adjusting the query.")
else:
    st.write("### Results")
    for i, (_, r) in enumerate(results_df.iterrows()):
        sc = scores[i] if (scores is not None and i < len(scores)) else None
        render_row(r, sc)
        st.divider()

st.write("### Curator Notes")
if "notes" not in st.session_state:
    st.session_state["notes"] = {} 

title_options = results_df["title"].tolist() if not results_df.empty else df["title"].tolist()
selected_title = st.selectbox("Select artwork by title", title_options, key="note_title_select")

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


def has_active_filters(yr_min, yr_max, year_range, artist_sel, medium_sel, dept_sel):
    year_filtered = not (year_range[0] == yr_min and year_range[1] == yr_max)
    return (
        year_filtered
        or artist_sel != "(any)"
        or medium_sel != "(any)"
        or dept_sel != "(any)"
    )

def current_analysis_df(df_all, results_df, query, yr_min, yr_max, year_range, artist_sel, medium_sel, dept_sel):
    if (query and query.strip()) or has_active_filters(yr_min, yr_max, year_range, artist_sel, medium_sel, dept_sel):
        return results_df
    return df_all

st.write("### Quick Analytics")


analysis_df = current_analysis_df(
    df_all=df,
    results_df=results_df,
    query=query,
    yr_min=yr_min,
    yr_max=yr_max,
    year_range=year_range,
    artist_sel=artist_sel,
    medium_sel=medium_sel,
    dept_sel=dept_sel,
)

if analysis_df.empty:
    st.info("No data to analyze for the current search/filters.")
else:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total works", len(analysis_df))
    with col2:
        st.metric("Artists (unique)", analysis_df["artist"].nunique())
    with col3:
        st.metric("Departments", analysis_df["department"].nunique())

    dept_counts = analysis_df["department"].value_counts().reset_index()
    dept_counts.columns = ["department", "count"]
    st.bar_chart(dept_counts.set_index("department"))


with st.expander("⚙️ Diagnostics"):
    st.write({
        "embedder_kind": embedder_kind,
        "sbert_available": sbert_embeddings is not None,
        "records": len(df),
        "hybrid_enabled": use_hybrid if 'use_hybrid' in locals() else None,
        "hybrid_weight": hybrid_w if 'hybrid_w' in locals() else None,
        "notes_count": len(st.session_state.get('notes', {})),
    })

