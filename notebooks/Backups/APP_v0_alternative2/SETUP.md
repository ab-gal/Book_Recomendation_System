# Book Recommendation App

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the app:**
   ```bash
   streamlit run app.py
   ```

## What's Included

- `app.py` - Main application with all search and recommendation engines
- `requirements.txt` - Python dependencies
- `README.md` - Detailed documentation

## Data Requirements

The app will automatically look for `books_merged_clean.csv` in:
- `../../data/clean/books_merged_clean.csv` (from notebooks folder)
- `../data/clean/books_merged_clean.csv`
- `data/clean/books_merged_clean.csv`
- `books_merged_clean.csv` (same folder)

## First Run

On first run, the app will:
1. Download the SBERT model (requires internet)
2. Generate embeddings (may take a few minutes)
3. Save embeddings as `book_embeddings.npy` for future use

Subsequent runs will be much faster!

## Features

- 🔍 **Dual Search**: Literal + semantic search
- 📖 **Smart Recommendations**: TF-IDF + SBERT hybrid
- 🎯 **Intelligent Prioritization**: Exact matches first, semantic results second
- 🌐 **Language & Year Filters**: Refine your search
- 📱 **Responsive UI**: Works on desktop and mobile
