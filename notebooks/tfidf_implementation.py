# Weighted TF-IDF Implementation for Book Recommendation System

# Import libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv(r"..\data\clean\books_merged_clean.csv")

# Check data structure
print("Data shape:", df.shape)
print("\nSample data:")
print(df[['title', 'author', 'language', 'subjects']].head(2))

# Clean and prepare text data
df['title_clean'] = df['title'].str.lower().fillna('')
df['subjects_clean'] = df['subjects'].str.lower().str.replace(', ', ' ').fillna('')
df['language_clean'] = df['language'].str.lower().fillna('')

# Create weighted combined text
# Title gets 3x weight, subjects get 1x weight, language gets 2x weight
df['weighted_text'] = (
    (df['title_clean'] + ' ') * 3 +  # Title repeated 3 times for higher weight
    df['subjects_clean'] + ' ' +     # Subjects normal weight
    (df['language_clean'] + ' ') * 2 # Language repeated 2 times for medium weight
)

print("\nSample weighted text:")
print(df['weighted_text'].iloc[0])

# Create TF-IDF matrix
tfidf = TfidfVectorizer(
    stop_words='english',
    max_features=5000,  # Limit features for performance
    ngram_range=(1, 2)  # Include bigrams for better context
)

tfidf_matrix = tfidf.fit_transform(df['weighted_text'])

print(f"\nTF-IDF matrix shape: {tfidf_matrix.shape}")

# Calculate cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

print(f"Similarity matrix shape: {cosine_sim.shape}")

# Function to get recommendations
def get_book_recommendations(title, df=df, cosine_sim=cosine_sim, top_n=5):
    """
    Get book recommendations based on title
    """
    # Find the index of the book that matches the title
    if title not in df['title'].values:
        return f"Book '{title}' not found in dataset"
    
    idx = df[df['title'] == title].index[0]
    
    # Get similarity scores for this book
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort books by similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get top N most similar books (excluding the book itself)
    sim_scores = sim_scores[1:top_n+1]
    
    # Get book indices
    book_indices = [i[0] for i in sim_scores]
    
    # Return top similar books
    recommendations = df.iloc[book_indices][['title', 'author', 'language', 'subjects']]
    recommendations['similarity_score'] = [i[1] for i in sim_scores]
    
    return recommendations

# Test the recommendation system
print("\n" + "="*50)
print("TESTING RECOMMENDATION SYSTEM")
print("="*50)

# Get a sample book title for testing
sample_book = df['title'].iloc[0]
print(f"\nFinding recommendations for: '{sample_book}'")
print("-" * 40)

recommendations = get_book_recommendations(sample_book)
print(recommendations)

# Test with another book
if len(df) > 100:
    sample_book_2 = df['title'].iloc[100]
    print(f"\nFinding recommendations for: '{sample_book_2}'")
    print("-" * 40)
    recommendations_2 = get_book_recommendations(sample_book_2)
    print(recommendations_2)

print("\n" + "="*50)
print("IMPLEMENTATION COMPLETE!")
print("="*50)
print("\nTo use the recommendation system:")
print("recommendations = get_book_recommendations('Your Book Title Here')")
