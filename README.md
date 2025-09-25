# Cleveland Museum of Art — AI-Assisted Analyzer (Prototype)

This Streamlit prototype demonstrates an AI-assisted analyzer for museum collections:
- **Hybrid retrieval**: semantic embeddings (SentenceTransformers) + TF-IDF fallback with a tunable weight.
- **Filters**: artist, department, medium, year range.
- **Curator helpers**: notes, CSV export, quick analytics.
- **Diagnostics**: see embedder mode and health.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
