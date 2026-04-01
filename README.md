# Book Recommendation System

## 🛠️ Tech Stack & Tools


![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![requests](https://img.shields.io/badge/requests-HTTP%20Client-2CA5E0?logo=python&logoColor=white)
![BeautifulSoup](https://img.shields.io/badge/BeautifulSoup-Web%20Scraping-4B8BBE?logo=python&logoColor=white)
![Open%20Library](https://img.shields.io/badge/Open%20Library-API-6B4E71?logo=bookstack&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App%20Deployment-FF4B4B?logo=streamlit&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

Overview
This project builds a book recommendation system using data collected from multiple sources. The goal is to demonstrate an end-to-end data analytics workflow, including web scraping, API integration, data cleaning, exploratory analysis, and deployment.

The final system allows users to explore books and receive recommendations through an interactive web application.

A comprehensive book recommendation app that combines both literal and semantic search engines with TF-IDF and SBERT recommendation systems.

## App

**🔗 Streamlit App**

[here](https://bookommender.streamlit.app)

Photo credit (https://www.pexels.com/de-de/foto/gestapelte-bucher-1333742/)

**🎥 Project Presentation**

[here](https://docs.google.com/presentation/d/1pwWplNvwnvt5cUMMA9SmXHMJTrSgokQhoSL6ocWgKiE/edit?usp=sharing)

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

## Data Sources


| Dataset                        | Source                                                    | Purpose                                                                                    |
| :----------------------------- | :-------------------------------------------------------- | :----------------------------------------------------------------------------------------- |
| **openlibrary**                | https://openlibrary.org/subjects/awards                   | Core data for books and awards given                                                       |





## Workflow

- Data collection (Scraping + API)

- Data cleaning & deduplication

- Exploratory analysis

- Content-based recommendation logic

- Deployment with Streamlit



## Future Improvements

- Add user-rating or popularity data

- Implement similarity using text descriptions (NLP)

- Improve genre standardisation

- Expand dataset beyond 1000 books

- Deploy using a cloud hosting platform
