# Cleveland Museum of Art Collection Explorer

This project is a Streamlit-based semantic search and collection analysis app for Cleveland Museum of Art collection data. It is designed to make artwork discovery more flexible than plain keyword lookup by combining semantic retrieval, keyword matching, metadata filters, and lightweight curator tools in one interface.

Instead of only searching for exact words, the app can surface artworks that are conceptually related to a query such as `bronze animal sculpture`, `religious manuscript`, or `portrait by a French artist`. The goal is to make the collection easier to explore for research, curation, and experimentation.

## What The App Does

- Performs semantic search over artwork records using SentenceTransformers embeddings when available
- Falls back to TF-IDF keyword retrieval when semantic models are unavailable
- Blends semantic and lexical retrieval in a hybrid ranking pipeline
- Optionally reranks top search candidates with a cross-encoder model for better precision
- Filters results by artist, medium, department, year range, open-access status, image availability, and gallery visibility
- Displays artwork details, images, tags, source links, and relevance scores
- Lets users save curator notes and export notes or result sets
- Provides quick analytics on the current result set

## Search Pipeline

The current search flow uses a layered retrieval approach:

1. The app builds a metadata-aware text representation for each artwork using fields like title, artist, year, medium, department, tags, and description.
2. It generates semantic embeddings for those records with a SentenceTransformers model when available.
3. It also builds a TF-IDF matrix as a lexical fallback and hybrid signal.
4. Query-time retrieval combines semantic similarity and TF-IDF scores.
5. The top candidates can then be reranked with a cross-encoder model for more accurate ordering.
6. Final ranking applies practical result cleanup such as deduplication and a few lightweight heuristics.

This gives the app a stronger ranking setup than a single keyword index while still preserving a dependable fallback path.

## Project Structure

- [app.py](/Users/eujinjeon/Cleveland_Museum_Database_Management/app.py) contains the Streamlit interface, data preparation, retrieval logic, reranking, filters, exports, and diagnostics
- [requirements.txt](/Users/eujinjeon/Cleveland_Museum_Database_Management/requirements.txt) contains the Python dependencies
- `data/data.csv` is the default collection dataset loaded by the app unless overridden with an environment variable

## Requirements

- Python 3.9 or newer is recommended
- A virtual environment is strongly recommended
- Internet access may be needed on first run if the embedding or reranker models are not already cached locally

## Installation

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `pip` points to a different Python installation on your machine, use:

```bash
python3 -m pip install -r requirements.txt
```

## Running The App

Basic run command:

```bash
streamlit run app.py
```

If `streamlit` is not found on your PATH, use:

```bash
python3 -m streamlit run app.py
```

Once the app starts, Streamlit will print a local URL, typically:

```text
http://localhost:8501
```

Open that URL in your browser to use the app.

## Configuration

The app supports a few environment variables so you can swap models or point to a different dataset without editing the code.

### Data Configuration

```bash
CMA_DATA_PATH=data/data.csv
CMA_MAX_ROWS=0
```

- `CMA_DATA_PATH` sets the CSV file to load
- `CMA_MAX_ROWS` optionally limits how many rows are loaded for faster testing; `0` means no limit

### Search Model Configuration

```bash
CMA_EMBED_MODEL=BAAI/bge-small-en-v1.5
CMA_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
CMA_RERANK_TOPN=50
```

- `CMA_EMBED_MODEL` chooses the SentenceTransformers embedding model
- `CMA_RERANKER_MODEL` chooses the cross-encoder reranker model
- `CMA_RERANK_TOPN` controls how many top candidates are reranked

Example run with explicit search model settings:

```bash
CMA_EMBED_MODEL=BAAI/bge-small-en-v1.5 \
CMA_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2 \
CMA_RERANK_TOPN=50 \
streamlit run app.py
```

## Using The Interface

After launching the app, you can:

- Enter a semantic query in the search box
- Adjust hybrid ranking weight between semantic and TF-IDF retrieval
- Filter by artist, medium, department, and year range
- Restrict results to CC0 works, works with images, or artworks currently on view
- Review ranked results with images, descriptions, and metadata
- Save curator notes for specific artworks
- Export current results or the full dataset with notes
- Inspect diagnostics to see which retrieval mode and models are active

## Notes On Model Behavior

- If SentenceTransformers is available and the embedding model loads successfully, the app uses semantic retrieval plus TF-IDF hybrid ranking.
- If semantic model loading fails, the app falls back to TF-IDF-only retrieval so the search interface still works.
- If the reranker model cannot be loaded, the app continues without reranking and reports that in the sidebar diagnostics.
- The first run may be slower because model files may need to download and cache locally.

## Why This Project Is Useful

Museum datasets often contain rich but uneven text fields. Exact keyword search can miss relevant works when descriptions, titles, or subject terms vary. A semantic search layer helps bridge those gaps by matching meaning, not just shared words. That makes the tool more useful for:

- collection exploration
- exhibition research
- subject-based browsing
- comparing related objects across departments or media
- testing retrieval ideas for digital humanities or museum informatics work

## Development Notes

- The app is implemented as a single Streamlit script for simplicity and ease of iteration
- Retrieval quality can be improved further by tuning score heuristics, building an evaluation set, or introducing persistent vector indexing
- Generated Python cache files are ignored via `.gitignore`

## Troubleshooting

If the app does not start:

- Make sure the virtual environment is activated
- Make sure dependencies installed successfully
- Try `python3 -m streamlit run app.py` instead of `streamlit run app.py`

If search quality seems weak:

- Confirm the semantic embedder loaded in the diagnostics panel
- Try a stronger embedding model
- Increase `CMA_RERANK_TOPN`
- Use broader natural-language queries rather than single keywords

If the dataset does not load:

- Confirm that the CSV exists at the path set by `CMA_DATA_PATH`
- Confirm the file has the expected collection columns used in [app.py](/Users/eujinjeon/Cleveland_Museum_Database_Management/app.py)

## Future Improvements

- Add persistent embedding storage instead of rebuilding every session
- Add a formal evaluation set for retrieval quality testing
- Introduce field-aware weighting for title, description, tags, and artist metadata
- Support approximate nearest-neighbor indexing for larger datasets
- Expand curator workflows with saved search views or collections
