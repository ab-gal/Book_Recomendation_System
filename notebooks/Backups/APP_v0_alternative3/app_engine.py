import pandas as pd
import numpy as np
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class BookAppEngine:
    def __init__(self, data_path, embeddings_path=None):
        print("Initializing Book App Engine...")
        self.df = pd.read_csv(data_path)
        self._preprocess_data()
        
        # --- Initialize Recommendation A (TF-IDF) ---
        print("Building TF-IDF model...")
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(self.df['soup'])
        self.cosine_sim_tfidf = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        self.indices = pd.Series(self.df.index, index=self.df['title']).drop_duplicates()
        
        # --- Initialize Search B / Recommendation B (SBERT) ---
        print("Loading SBERT model...")
        self.sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        if embeddings_path and os.path.exists(embeddings_path):
            print(f"Loading embeddings from {embeddings_path}...")
            self.sbert_embeddings = np.load(embeddings_path)
        else:
            print("Computing SBERT embeddings (this might take a while)...")
            self.sbert_embeddings = self.sbert_model.encode(
                self.df["content"].tolist(),
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            # Save for future use if a path was provided
            if embeddings_path:
                np.save(embeddings_path, self.sbert_embeddings)
                
        print("Engine Ready! ✅")

    def _preprocess_data(self):
        # Fill NaNs
        self.df['title'] = self.df['title'].fillna('')
        self.df['author'] = self.df['author'].fillna('')
        self.df['subjects'] = self.df['subjects'].fillna('')
        self.df['language'] = self.df['language'].fillna('')
        self.df['published_year'] = self.df['published_year'].fillna(0)

        # Create 'soup' for TF-IDF (Weighted: Author x3, Title x2, Language x2, Subjects x1)
        self.df['soup'] = self.df['author'].apply(lambda x: ' '.join([x,x,x])) + ' ' + \
                          self.df['title'].apply(lambda x: ' '.join([x,x])) + ' ' + \
                          self.df['language'].apply(lambda x: ' '.join([x,x])) + ' ' + \
                          self.df['subjects']
        
        # Create 'content' for SBERT
        def clean_text(s):
            s = str(s).lower().strip()
            s = re.sub(r"\s+", " ", s)
            s = s.replace(",", " ")
            return s

        self.df["content"] = (
            self.df["title"].apply(clean_text) + " [SEP] " +
            self.df["author"].apply(clean_text) + " [SEP] " +
            self.df["subjects"].apply(clean_text)
        )

    # ---------------------------------------------------------
    # SEARCH ENGINE SYSTEM
    # ---------------------------------------------------------
    
    def search_A(self, keyword):
        """
        Search A: Literal search in 'soup' (Title, Author, Language, Subjects).
        """
        results = self.df[self.df['soup'].str.contains(keyword, case=False, na=False)]
        return results

    def search_B(self, keyword, top_k=10):
        """
        Search B: Semantic search using SBERT.
        """
        query_vec = self.sbert_model.encode([keyword], normalize_embeddings=True)
        sims = cosine_similarity(query_vec, self.sbert_embeddings).flatten()
        top_idx = sims.argsort()[::-1][:top_k]
        
        results = self.df.iloc[top_idx].copy()
        results['similarity'] = sims[top_idx]
        return results

    def hybrid_search(self, keyword, total_results=10):
        """
        The System: Search A first, then Search B.
        """
        # 1. Get results from Search A
        results_A = self.search_A(keyword)
        
        # 2. Get results from Search B
        # We ask for more than we need to fill the gap
        results_B = self.search_B(keyword, top_k=total_results)
        
        # 3. Combine: A first, then B (excluding duplicates)
        combined_results = pd.concat([results_A, results_B])
        
        # Remove duplicates based on index (keeping the first occurrence, which is from A)
        combined_results = combined_results[~combined_results.index.duplicated(keep='first')]
        
        # Limit to total_results
        return combined_results.head(total_results)[['title', 'author', 'published_year', 'language', 'subjects']]

    # ---------------------------------------------------------
    # RECOMMENDATION ENGINE SYSTEM
    # ---------------------------------------------------------

    def recommend_A(self, title, top_k=10):
        """
        Recommendation A: TF-IDF + Cosine Similarity (Weighted).
        """
        if title not in self.indices:
            return pd.DataFrame() # Return empty if not found

        idx = self.indices[title]
        
        # Handle case where there are duplicate titles (take the first one)
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]

        sim_scores = list(enumerate(self.cosine_sim_tfidf[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_k+1] # Exclude self
        
        book_indices = [i[0] for i in sim_scores]
        results = self.df.iloc[book_indices].copy()
        results['similarity_score'] = [i[1] for i in sim_scores]
        results['source'] = 'Rec A (TF-IDF)'
        return results

    def recommend_B(self, title, top_k=10):
        """
        Recommendation B: SBERT Semantic Similarity.
        """
        # Find the book's embedding
        # We need the index of the book in the dataframe
        matches = self.df[self.df['title'] == title]
        if matches.empty:
            return pd.DataFrame()
        
        idx = matches.index[0]
        book_vec = self.sbert_embeddings[idx].reshape(1, -1)
        
        sims = cosine_similarity(book_vec, self.sbert_embeddings).flatten()
        
        # Get top results (excluding self)
        # We get top_k + 1 because the book itself will be the top match
        top_idx = sims.argsort()[::-1][1:top_k+1]
        
        results = self.df.iloc[top_idx].copy()
        results['similarity_score'] = sims[top_idx]
        results['source'] = 'Rec B (SBERT)'
        return results

    def hybrid_recommend(self, title, total_results=10):
        """
        The System: Rec A first, then Rec B.
        """
        # 1. Get results from Rec A
        results_A = self.recommend_A(title, top_k=total_results)
        
        # 2. Get results from Rec B
        results_B = self.recommend_B(title, top_k=total_results)
        
        # 3. Combine: A first, then B
        combined_results = pd.concat([results_A, results_B])
        
        # Remove duplicates (keeping A's version)
        combined_results = combined_results[~combined_results.index.duplicated(keep='first')]
        
        return combined_results.head(total_results)[['title', 'author', 'published_year', 'language', 'subjects', 'source']]

# ---------------------------------------------------------
# APP SIMULATION (CLI)
# ---------------------------------------------------------

def main():
    # Paths
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, "../../data/clean/books_merged_clean.csv")
    embeddings_path = os.path.join(base_path, "book_embeddings.npy")
    
    # Initialize Engine
    app = BookAppEngine(data_path, embeddings_path)
    
    print("\n" + "="*50)
    print("      WELCOME TO THE BOOK RECOMMENDATION APP      ")
    print("="*50 + "\n")
    
    while True:
        # Step 1: Main Page / Search Trigger
        query = input("Step 1: Enter a keyword to search (or 'exit' to quit): ").strip()
        if query.lower() == 'exit':
            break
        
        if not query:
            continue

        # Step 2: Search Results
        print(f"\nSearching for '{query}'...")
        search_results = app.hybrid_search(query)
        
        if search_results.empty:
            print("No books found. Try again.")
            continue
            
        print("\n--- Search Results ---")
        # Display with an index for selection
        display_df = search_results.reset_index(drop=True)
        print(display_df[['title', 'author', 'published_year']])
        
        print("\nOptions:")
        print(" - Type a number (0-9) to select a book and see recommendations.")
        print(" - Type any text to search again.")
        
        user_input = input("Your choice: ").strip()
        
        # Check if user selected a book (Step 3.2) or wants to search again (Step 3.1)
        if user_input.isdigit():
            selection_idx = int(user_input)
            if 0 <= selection_idx < len(display_df):
                selected_book = display_df.iloc[selection_idx]
                book_title = selected_book['title']
                
                print(f"\n" + "-"*30)
                print(f"SELECTED BOOK: {book_title}")
                print(f"Author: {selected_book['author']}")
                print(f"Year: {selected_book['published_year']}")
                print("-"*30)
                
                print(f"\nGenerating recommendations for '{book_title}'...")
                recommendations = app.hybrid_recommend(book_title)
                
                print("\n--- Recommendations ---")
                print(recommendations)
                print("\n(Press Enter to return to search)")
                input()
            else:
                print("Invalid selection. Returning to search.")
        else:
            # User typed text, treat as new search (Loop back to Step 1 logic)
            # In this loop structure, we just continue, and the next iteration asks for input
            print(f"Starting new search for '{user_input}'...")
            # We can either loop back to the top or handle the search immediately.
            # To keep the flow simple, let's just loop back.
            pass

if __name__ == "__main__":
    main()
