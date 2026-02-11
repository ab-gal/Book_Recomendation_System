# Book Recommendation System

A comprehensive book recommendation app that combines both literal and semantic search engines with TF-IDF and SBERT recommendation systems.

## Features

### Search Engine System
- **Search A (Literal)**: Direct keyword matching across all book fields
- **Search B (Semantic)**: SBERT-based semantic understanding
- **Combined System**: Literal results first, then semantic results

### Recommendation Engine System  
- **Recommendation A (TF-IDF)**: Weighted feature similarity (Author 3x, Title 2x, Subjects 2x, Language 1x)
- **Recommendation B (SBERT)**: Semantic similarity understanding
- **Combined System**: TF-IDF results first, then SBERT recommendations

### User Flow
1. **Main Page**: Search interface with filters
2. **Search Results**: Combined search results with book selection
3. **Book Details + Recommendations**: Selected book with personalized recommendations

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the app:
```bash
streamlit run app.py
```

## Data Setup

The app expects the book dataset at one of these locations:
- `../../data/clean/books_merged_clean.csv`
- `../data/clean/books_merged_clean.csv`
- `data/clean/books_merged_clean.csv`
- `books_merged_clean.csv`

On first run, the app will generate SBERT embeddings and save them as `book_embeddings.npy` for faster subsequent startups.

## Usage

1. **Search**: Type any query (title, author, topic, keyword)
2. **Filter**: Use language and year filters in sidebar
3. **Explore**: Click on books to see detailed recommendations
4. **Navigate**: Use sidebar buttons to switch between views

## Architecture

- **Frontend**: Streamlit web interface
- **Search**: Combined literal + semantic search
- **Recommendations**: TF-IDF + SBERT hybrid approach
- **Caching**: Efficient model and data loading
- **Error Handling**: Robust error messages and fallbacks
