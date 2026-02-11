import streamlit as st
import pandas as pd
import os
from app_engine import BookAppEngine

# --- Page Configuration ---
st.set_page_config(
    page_title="Book Recommendation App",
    page_icon="📚",
    layout="wide"
)

# --- Caching the Engine ---
@st.cache_resource
def load_engine():
    """
    Loads the BookAppEngine and caches it to avoid reloading models on every interaction.
    """
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, "../../data/clean/books_merged_clean.csv")
    embeddings_path = os.path.join(base_path, "book_embeddings.npy")
    
    engine = BookAppEngine(data_path, embeddings_path)
    return engine

# --- Main App Logic ---
def main():
    st.title("📚 Book Search & Recommendation Engine")

    # Load the engine
    engine = load_engine()

    # --- Initialize Session State ---
    # This is crucial for remembering the app's state between interactions.
    if 'view' not in st.session_state:
        st.session_state.view = 'search'  # 'search' or 'recommend'
    if 'selected_book' not in st.session_state:
        st.session_state.selected_book = None
    if 'search_query' not in st.session_state:
        st.session_state.search_query = ""

    # --- Search Bar ---
    # This is the main trigger for the app (Step 1)
    search_query = st.text_input(
        "Search for a book by title, author, or topic...",
        placeholder="e.g., 'history', 'J.K. Rowling', or 'The Great Gatsby'",
        key="search_bar"
    )

    # If a new search is performed, reset to search view
    if search_query and search_query != st.session_state.search_query:
        st.session_state.search_query = search_query
        st.session_state.view = 'search'
        st.session_state.selected_book = None

    # --- View Controller ---
    if st.session_state.view == 'search':
        handle_search_view(engine, search_query)
    elif st.session_state.view == 'recommend':
        handle_recommend_view(engine)

def handle_search_view(engine, query):
    """
    Displays the search results (Step 2 & 3.1).
    """
    if not query:
        st.info("Welcome! Type something in the search bar above to find books.")
        return

    st.header("Search Results")
    st.markdown(f"Showing results for: **'{query}'**")
    
    search_results = engine.hybrid_search(query)

    if search_results.empty:
        st.warning("No books found. Please try another search.")
        return

    # Display results and handle selection
    for i, row in search_results.iterrows():
        st.divider()
        col1, col2 = st.columns([4, 1])
        with col1:
            st.subheader(row['title'])
            st.caption(f"**Author:** {row['author']} | **Year:** {int(row['published_year'])} | **Language:** {row['language'].capitalize()}")
            # Truncate long subjects for cleaner display
            subjects = row['subjects']
            if isinstance(subjects, str) and len(subjects) > 200:
                subjects = subjects[:200] + "..."
            st.write(subjects)
        with col2:
            # When this button is clicked, it will set the state and rerun the app
            if st.button("Get Recommendations", key=f"rec_{i}"):
                st.session_state.view = 'recommend'
                st.session_state.selected_book = row
                st.rerun() # Immediately rerun to switch to the recommendation view

def handle_recommend_view(engine):
    """
    Displays the selected book and its recommendations (Step 3.2).
    """
    selected_book = st.session_state.selected_book
    
    if selected_book is None:
        st.error("Something went wrong. Please search again.")
        # Button to reset state
        if st.button("Back to Search"):
            st.session_state.view = 'search'
            st.session_state.selected_book = None
            st.rerun()
        return

    # --- Back Button ---
    if st.button("← Back to Search Results"):
        st.session_state.view = 'search'
        st.session_state.selected_book = None
        st.rerun()

    # --- Display Selected Book ---
    st.header(f"Recommendations for: {selected_book['title']}")
    st.subheader(f"by {selected_book['author']}")
    st.markdown(f"**Year:** {int(selected_book['published_year'])} | **Language:** {selected_book['language'].capitalize()}")
    st.markdown(f"**Subjects:** {selected_book['subjects']}")
    st.divider()

    # --- Get and Display Recommendations ---
    recommendations = engine.hybrid_recommend(selected_book['title'])

    if recommendations.empty:
        st.warning("Could not find any recommendations for this book.")
    else:
        st.subheader("Top Recommendations")
        # Display as a clean table
        st.dataframe(
            recommendations,
            use_container_width=True,
            column_config={
                "title": "Title",
                "author": "Author",
                "published_year": "Year",
                "language": "Language",
                "subjects": "Subjects",
                "source": st.column_config.TextColumn( # Replaced TagColumn with TextColumn for compatibility
                    "Source",
                    help="**Rec A (TF-IDF):** More literal match.\n\n**Rec B (SBERT):** More semantic match."
                )
            },
            hide_index=True
        )

if __name__ == "__main__":
    main()
