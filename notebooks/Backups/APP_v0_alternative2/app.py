import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ----------------------------
# Configuration and Setup
# ----------------------------
st.set_page_config(
    page_title="📚 Book Recommendation System", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Data Loading and Caching
# ----------------------------
@st.cache_resource
def load_data():
    """Load the book dataset"""
    # Try to find the data file in different possible locations
    data_paths = [
        "../../data/clean/books_merged_clean.csv",
        "../data/clean/books_merged_clean.csv", 
        "data/clean/books_merged_clean.csv",
        "books_merged_clean.csv"
    ]
    
    for path in data_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            break
    else:
        st.error("Could not find books_merged_clean.csv. Please ensure the data file is in the correct location.")
        st.stop()
    
    # Fill NaN values for text processing
    df['title'] = df['title'].fillna('')
    df['author'] = df['author'].fillna('')
    df['subjects'] = df['subjects'].fillna('')
    df['language'] = df['language'].fillna('')
    
    return df

def load_tfidf_components(df_soup_series):
    """Load and cache TF-IDF components"""
    # Initialize TF-IDF Vectorizer
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_soup_series)
    cosine_sim_tfidf = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return tfidf, tfidf_matrix, cosine_sim_tfidf

@st.cache_resource
def load_sbert_model():
    """Load SBERT model"""
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"Error loading SBERT model: {e}")
        st.error("Please ensure you have an internet connection and the required packages installed.")
        st.stop()

@st.cache_resource
def load_or_create_sbert_embeddings(_model, content_list):
    """Load existing embeddings or create new ones"""
    embeddings_path = "book_embeddings.npy"
    
    if os.path.exists(embeddings_path):
        try:
            return np.load(embeddings_path)
        except Exception as e:
            st.warning(f"Could not load existing embeddings: {e}. Creating new ones...")
    
    # Generate embeddings
    with st.spinner("Generating SBERT embeddings (this may take a few minutes on first run)..."):
        try:
            embeddings = _model.encode(
                content_list,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            # Save embeddings for future use
            np.save(embeddings_path, embeddings)
            st.success("Embeddings generated and saved successfully!")
            return embeddings
        except Exception as e:
            st.error(f"Error generating embeddings: {e}")
            st.stop()

# ----------------------------
# Search Engine A: Literal Search (TF-IDF based)
# ----------------------------
def search_books_literal(df, keyword, max_results=10):
    """
    Search A: More straightforward and literal search using keyword matching
    """
    # Search for the keyword (case-insensitive) in all relevant columns
    mask = (
        df['title'].str.contains(keyword, case=False, na=False) |
        df['author'].str.contains(keyword, case=False, na=False) |
        df['subjects'].str.contains(keyword, case=False, na=False) |
        df['language'].str.contains(keyword, case=False, na=False)
    )
    
    results = df[mask].copy()
    
    if results.empty:
        return pd.DataFrame()
    
    # Return results with similarity score (1.0 for exact matches)
    results['similarity'] = 1.0
    results['search_type'] = 'Literal'
    
    return results[['title', 'author', 'published_year', 'language', 'subjects', 'cover', 'similarity', 'search_type']].head(max_results)

# ----------------------------
# Search Engine B: Semantic Search (SBERT)
# ----------------------------
def search_books_semantic(df, _model, embeddings, query, max_results=10):
    """
    Search B: Semantic search using SBERT
    """
    def clean_text(s):
        s = str(s).lower().strip()
        s = re.sub(r"\s+", " ", s)
        s = s.replace(",", " ")
        return s
    
    query_clean = clean_text(query)
    q_vec = _model.encode([query_clean], normalize_embeddings=True)
    
    # Calculate cosine similarity
    sims = cosine_similarity(q_vec, embeddings).flatten()
    
    # Get top results
    top_idx = sims.argsort()[::-1][:max_results]
    
    results = df.loc[top_idx].copy()
    results['similarity'] = sims[top_idx]
    results['search_type'] = 'Semantic'
    
    return results[['title', 'author', 'published_year', 'language', 'subjects', 'cover', 'similarity', 'search_type']]

# ----------------------------
# Combined Search System
# ----------------------------
def combined_search(df, _model, embeddings, query, max_total_results=20):
    """
    Combined search system: Search A results first, then Search B
    """
    # Search A: Literal search (get all matches, but cap at total desired results)
    literal_results = search_books_literal(df, query, max_results=max_total_results)
    
    # Search B: Semantic search (fill remaining slots)
    remaining_slots = max_total_results - len(literal_results)
    if remaining_slots > 0:
        semantic_results = search_books_semantic(df, _model, embeddings, query, max_results=remaining_slots + 10)  # Get extra to filter out duplicates
        
        # Remove books that are already in literal results
        if not semantic_results.empty:
            semantic_results = semantic_results[~semantic_results['title'].isin(literal_results['title'])]
            semantic_results = semantic_results.head(remaining_slots)
    else:
        semantic_results = pd.DataFrame()
    
    # Combine results
    if literal_results.empty:
        final_results = semantic_results.head(max_total_results)
    else:
        final_results = pd.concat([literal_results, semantic_results.head(remaining_slots)], ignore_index=True)
    
    # Reset index to start from 1
    if not final_results.empty:
        final_results = final_results.reset_index(drop=True)
        final_results.index = final_results.index + 1
    
    return final_results

# ----------------------------
# Recommendation Engine A: TF-IDF Recommendations
# ----------------------------
def recommend_tfidf(df, title, cosine_sim_tfidf, indices, top_k=10):
    """
    Recommendation A: TF-IDF based recommendations
    """
    # Get the index of the book that matches the title
    if title not in indices:
        possible_titles = [t for t in indices.index if title.lower() in t.lower()]
        if not possible_titles:
            return pd.DataFrame()
        title = possible_titles[0]
    
    idx = indices[title]
    
    # Get the pairwise similarity scores
    sim_scores = list(enumerate(cosine_sim_tfidf[idx]))
    
    # Sort the books based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores of the top similar books (excluding the book itself)
    sim_scores = sim_scores[1:top_k+1]
    
    # Get the book indices and similarity scores
    book_indices = [i[0] for i in sim_scores]
    similarity_scores = [i[1] for i in sim_scores]
    
    # Get the top similar books
    recommendations_df = df.iloc[book_indices].copy()
    recommendations_df['similarity'] = similarity_scores
    recommendations_df['recommendation_type'] = 'TF-IDF'
    
    return recommendations_df[['title', 'author', 'published_year', 'language', 'subjects', 'cover', 'similarity', 'recommendation_type']]

# ----------------------------
# Recommendation Engine B: SBERT Recommendations
# ----------------------------
def recommend_sbert(df, _model, embeddings, title, top_k=10):
    """
    Recommendation B: SBERT based recommendations
    """
    # Find the book index
    book_idx = df[df['title'] == title].index
    if len(book_idx) == 0:
        # Try partial match
        possible_titles = df[df['title'].str.contains(title, case=False, na=False)]
        if possible_titles.empty:
            return pd.DataFrame()
        book_idx = [possible_titles.index[0]]
        title = possible_titles.iloc[0]['title']
    
    book_idx = book_idx[0]
    
    # Get the book's embedding
    book_embedding = embeddings[book_idx:book_idx+1]
    
    # Calculate cosine similarity with all books
    sims = cosine_similarity(book_embedding, embeddings).flatten()
    
    # Get top similar books (excluding the book itself)
    sims[book_idx] = 0  # Exclude the book itself
    top_idx = sims.argsort()[::-1][:top_k]
    
    recommendations_df = df.iloc[top_idx].copy()
    recommendations_df['similarity'] = sims[top_idx]
    recommendations_df['recommendation_type'] = 'SBERT'
    
    return recommendations_df[['title', 'author', 'published_year', 'language', 'subjects', 'cover', 'similarity', 'recommendation_type']]

# ----------------------------
# Combined Recommendation System
# ----------------------------
def combined_recommendations(df, _model, embeddings, title, cosine_sim_tfidf, indices, top_k=10):
    """
    Combined recommendation system: Recommendation A results first, then Recommendation B
    """
    # Recommendation A: TF-IDF (max 6 results)
    tfidf_recs = recommend_tfidf(df, title, cosine_sim_tfidf, indices, top_k=6)
    
    # Recommendation B: SBERT (fill remaining slots)
    remaining_slots = top_k - len(tfidf_recs)
    sbert_recs = recommend_sbert(df, _model, embeddings, title, top_k=remaining_slots + 5)  # Get extra to filter out duplicates
    
    # Remove books that are already in TF-IDF recommendations
    if not tfidf_recs.empty and not sbert_recs.empty:
        sbert_recs = sbert_recs[~sbert_recs['title'].isin(tfidf_recs['title'])]
    
    # Combine results
    if tfidf_recs.empty:
        final_recs = sbert_recs.head(top_k)
    else:
        final_recs = pd.concat([tfidf_recs, sbert_recs.head(remaining_slots)], ignore_index=True)
    
    # Reset index to start from 1
    if not final_recs.empty:
        final_recs = final_recs.reset_index(drop=True)
        final_recs.index = final_recs.index + 1
    
    return final_recs

# ----------------------------
# UI Helper Functions
# ----------------------------
def display_book_card(book_data, show_search_type=False, show_recommendation_type=False):
    """Display a single book as a card"""
    with st.container(border=True):
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if isinstance(book_data.get("cover"), str) and book_data["cover"].startswith("http"):
                st.image(book_data["cover"], width=120)
            else:
                st.image("https://openlibrary.org/images/icons/avatar_book.png", width=120)
        
        with col2:
            st.markdown(f"### {book_data['title']}")
            st.markdown(f"**Author:** {book_data['author']}")
            st.markdown(f"**Year:** {book_data['published_year']}  |  **Language:** {book_data['language']}")
            
            if show_search_type and 'search_type' in book_data:
                st.markdown(f"**Search Type:** {book_data['search_type']}")
            
            if show_recommendation_type and 'recommendation_type' in book_data:
                st.markdown(f"**Recommendation Type:** {book_data['recommendation_type']}")
            
            st.markdown(f"**Similarity:** {book_data['similarity']:.3f}")
            st.markdown(f"**Subjects:** {book_data['subjects']}")
            
            # Add clickable button for this book
            if st.button(f"📖 View Details", key=f"book_{book_data['title']}_{hash(str(book_data))}"):
                st.session_state.selected_book = book_data
                st.session_state.current_view = 'recommendations'
                st.rerun()

def display_selected_book(book_data):
    """Display the selected book in detail"""
    st.markdown("---")
    st.markdown("## 📖 Selected Book Details")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if isinstance(book_data.get("cover"), str) and book_data["cover"].startswith("http"):
            st.image(book_data["cover"], width=200)
        else:
            st.image("https://openlibrary.org/images/icons/avatar_book.png", width=200)
    
    with col2:
        st.markdown(f"### {book_data['title']}")
        st.markdown(f"**Author:** {book_data['author']}")
        st.markdown(f"**Published Year:** {book_data['published_year']}")
        st.markdown(f"**Language:** {book_data['language']}")
        st.markdown(f"**Subjects:** {book_data['subjects']}")
        
        if isinstance(book_data.get("cover"), str) and book_data["cover"].startswith("http"):
            st.markdown(f"**Cover Link:** [{book_data['cover']}]({book_data['cover']})")

# ----------------------------
# Main Application
# ----------------------------
def main():
    # Initialize session state
    if 'current_view' not in st.session_state:
        st.session_state.current_view = 'search'
    if 'selected_book' not in st.session_state:
        st.session_state.selected_book = None
    if 'last_search_query' not in st.session_state:
        st.session_state.last_search_query = ""
    
    try:
        # Load data and models
        df = load_data()
        
        # Create TF-IDF soup (not cached, as it modifies DataFrame)
        df['soup'] = (
            df['author'].apply(lambda x: ' '.join([x, x, x])) + ' ' +  # Author: 3x weight
            df['title'].apply(lambda x: ' '.join([x, x])) + ' ' +       # Title: 2x weight  
            df['subjects'].apply(lambda x: ' '.join([x, x])) + ' ' +     # Subjects: 2x weight (topics/themes)
            df['language']                                                # Language: 1x weight (filter, not similarity)
        )
        
        # Create title index mapping
        indices = pd.Series(df.index, index=df['title']).drop_duplicates()
        
        # Load TF-IDF components (cached)
        tfidf, tfidf_matrix, cosine_sim_tfidf = load_tfidf_components(df['soup'])
        
        # Load SBERT model and embeddings (cached)
        model = load_sbert_model()
        
        # Create SBERT content (not cached, as it modifies DataFrame)
        def clean_text(s):
            s = str(s).lower().strip()
            s = re.sub(r"\s+", " ", s)
            s = s.replace(",", " ")
            return s
        
        df["content"] = (
            df["title"].apply(clean_text) + " [SEP] " +
            df["author"].apply(clean_text) + " [SEP] " +
            df["subjects"].apply(clean_text)
        )
        
        embeddings = load_or_create_sbert_embeddings(model, df["content"].tolist())
    except Exception as e:
        st.error(f"Error initializing application: {e}")
        st.stop()
    
    # Main title
    st.title("📚 Book Recommendation System")
    st.markdown("Discover your next favorite book using our advanced search and recommendation engines!")
    
    # Sidebar for navigation and filters
    with st.sidebar:
        st.header("🔍 Navigation & Filters")
        
        # Navigation buttons
        if st.button("🏠 Home", use_container_width=True):
            st.session_state.current_view = 'search'
            st.session_state.selected_book = None
            st.rerun()
        
        if st.session_state.selected_book:
            if st.button("📖 Current Selection", use_container_width=True):
                st.session_state.current_view = 'recommendations'
                st.rerun()
        
        st.markdown("---")
        
        # Search filters
        st.subheader("Search Filters")
        max_results = st.slider("Max Results", min_value=5, max_value=20, value=10, step=1)
        
        languages = ["All"] + sorted(df["language"].dropna().astype(str).str.strip().str.lower().unique().tolist())
        selected_language = st.selectbox("Language", languages, index=0)
        
        min_year = int(df["published_year"].min())
        max_year = int(df["published_year"].max())
        year_range = st.slider("Publication Year Range", min_value=min_year, max_value=max_year, value=(min_year, max_year))
    
    # Main content area based on current view
    if st.session_state.current_view == 'search':
        # Step 1: Main page - Search interface
        st.markdown("## 🔍 Search for Books")
        st.markdown("Type what you're looking for - title, author, topic, or any keyword!")
        
        # Search input
        search_query = st.text_input(
            "Search Query",
            placeholder="e.g., modern architecture, baseball, award-winning books...",
            value=st.session_state.last_search_query
        )
        
        # Search button
        if st.button("🔍 Search", use_container_width=True) or (search_query and search_query != st.session_state.last_search_query):
            if search_query.strip():
                st.session_state.last_search_query = search_query
                with st.spinner("Searching..."):
                    # Perform combined search
                    results = combined_search(df, model, embeddings, search_query, max_results)
                    
                    # Apply filters
                    if selected_language != "All":
                        results = results[results['language'].str.lower() == selected_language.lower()]
                    
                    if not results.empty:
                        year_min_selected, year_max_selected = year_range
                        results = results[
                            (results['published_year'] >= year_min_selected) & 
                            (results['published_year'] <= year_max_selected)
                        ]
                    
                    st.session_state.search_results = results
                    st.session_state.current_view = 'search_results'
                    st.rerun()
            else:
                st.warning("Please enter a search query.")
        
        # Display some sample books if no search yet
        if not st.session_state.last_search_query:
            st.markdown("### 📚 Sample Books")
            sample_books = df.sample(5, random_state=42)
            for _, book in sample_books.iterrows():
                display_book_card(book.to_dict())
    
    elif st.session_state.current_view == 'search_results':
        # Step 2: Search results
        st.markdown(f"## 🔍 Search Results for: '{st.session_state.last_search_query}'")
        
        if 'search_results' in st.session_state and not st.session_state.search_results.empty:
            results = st.session_state.search_results
            
            # Apply filters again in case they changed
            if selected_language != "All":
                results = results[results['language'].str.lower() == selected_language.lower()]
            
            year_min_selected, year_max_selected = year_range
            results = results[
                (results['published_year'] >= year_min_selected) & 
                (results['published_year'] <= year_max_selected)
            ]
            
            if not results.empty:
                st.success(f"Found {len(results)} books!")
                
                # Display results
                for idx, (_, book) in enumerate(results.iterrows()):
                    display_book_card(book.to_dict(), show_search_type=True)
                
                # New search input at bottom
                st.markdown("---")
                st.markdown("### 🔍 Search Again")
                new_search_query = st.text_input(
                    "New Search Query",
                    placeholder="Try another search...",
                    key="new_search"
                )
                
                if st.button("🔍 Search Again", use_container_width=True) or new_search_query:
                    if new_search_query.strip():
                        st.session_state.last_search_query = new_search_query
                        with st.spinner("Searching..."):
                            new_results = combined_search(df, model, embeddings, new_search_query, max_results)
                            st.session_state.search_results = new_results
                            st.rerun()
                    elif new_search_query == "":
                        st.warning("Please enter a search query.")
            else:
                st.warning("No results found with current filters. Try adjusting your filters.")
                if st.button("Clear Filters"):
                    st.rerun()
        else:
            st.error("No search results found.")
            if st.button("🔍 New Search"):
                st.session_state.current_view = 'search'
                st.session_state.search_results = None
                st.rerun()
    
    elif st.session_state.current_view == 'recommendations':
        # Step 3.2: Book details and recommendations
        if st.session_state.selected_book:
            # Display selected book
            display_selected_book(st.session_state.selected_book)
            
            # Get recommendations
            with st.spinner("Generating recommendations..."):
                recommendations = combined_recommendations(
                    df, model, embeddings, 
                    st.session_state.selected_book['title'], 
                    cosine_sim_tfidf, indices, 
                    top_k=10
                )
            
            st.markdown("---")
            st.markdown("## 📖 Recommended Books")
            
            if not recommendations.empty:
                st.success(f"Found {len(recommendations)} recommendations!")
                
                for idx, (_, book) in enumerate(recommendations.iterrows()):
                    display_book_card(book.to_dict(), show_recommendation_type=True)
            else:
                st.warning("No recommendations found for this book.")
            
            # Action buttons
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔍 New Search", use_container_width=True):
                    st.session_state.current_view = 'search'
                    st.session_state.selected_book = None
                    st.session_state.search_results = None
                    st.session_state.last_search_query = ""
                    st.rerun()
            
            with col2:
                if st.button("📚 Back to Results", use_container_width=True):
                    st.session_state.current_view = 'search_results'
                    st.rerun()
        else:
            st.error("No book selected. Please search for a book first.")
            if st.button("🔍 Go to Search"):
                st.session_state.current_view = 'search'
                st.rerun()

if __name__ == "__main__":
    main()
