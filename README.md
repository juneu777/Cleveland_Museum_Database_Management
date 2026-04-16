# Cleveland Museum of Art — AI-Assisted Analyzer (Prototype)

This Streamlit prototype demonstrates an AI-assisted analyzer for museum collections:
- **Hybrid retrieval**: semantic embeddings (SentenceTransformers) + TF-IDF fallback with a tunable weight.
- **Better ranking**: richer metadata-aware indexing with optional cross-encoder reranking of top candidates.
- **Filters**: artist, department, medium, year range.
- **Curator helpers**: notes, CSV export, quick analytics.
- **Diagnostics**: see embedder mode and health.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py

# Optional model overrides
CMA_EMBED_MODEL=BAAI/bge-small-en-v1.5 \
CMA_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2 \
CMA_RERANK_TOPN=50 \
streamlit run app.py
