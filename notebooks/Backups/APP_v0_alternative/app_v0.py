import numpy as np
import pandas as pd
import re
import streamlit as st

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# Helpers
# ----------------------------
def clean_text(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace(",", " ")
    return s

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def load_data():
    df = pd.read_csv("books_with_content.csv")
    emb = np.load("book_embeddings.npy")
    return df, emb

def recommend_filtered(df, embeddings, model, query, top_k=10, language=None, min_year=None, max_year=None):
    query = clean_text(query)
    q_vec = model.encode([query], normalize_embeddings=True)

    sims = cosine_similarity(q_vec, embeddings).flatten()
    df_sim = df.copy()
    df_sim["similarity"] = sims

    if language and language != "All":
        df_sim = df_sim[df_sim["language"].str.lower() == language.lower()]

    if min_year is not None:
        df_sim = df_sim[df_sim["published_year"] >= min_year]

    if max_year is not None:
        df_sim = df_sim[df_sim["published_year"] <= max_year]

    results = (
        df_sim.sort_values("similarity", ascending=False)
        .head(top_k)[["title", "author", "published_year", "language", "subjects", "cover", "similarity"]]
        .reset_index(drop=True)
    )
    results.index = results.index + 1  # 1..top_k ranking
    return results

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Book Recommender (SBERT)", layout="wide")
st.title("📚 Book Recommender (SBERT Semantic Search)")
st.write("Type what you’re looking for (topic, author, style). Results are ranked by semantic similarity.")

df, embeddings = load_data()
model = load_model()

# Sidebar filters
st.sidebar.header("Filters")
top_k = st.sidebar.slider("Number of recommendations", min_value=3, max_value=20, value=10, step=1)

languages = ["All"] + sorted(df["language"].dropna().astype(str).str.strip().str.lower().unique().tolist())
language = st.sidebar.selectbox("Language", languages, index=0)

min_year = int(df["published_year"].min())
max_year = int(df["published_year"].max())
year_range = st.sidebar.slider("Published year range", min_value=min_year, max_value=max_year, value=(min_year, max_year))
year_min_selected, year_max_selected = year_range

query = st.text_input("Search query", placeholder="e.g., modern architecture award winning houses, olympic sports champions, digital art...")

if st.button("Recommend") or (query and len(query.strip()) > 0):
    if not query.strip():
        st.warning("Please type a search query.")
    else:
        results = recommend_filtered(
            df=df,
            embeddings=embeddings,
            model=model,
            query=query,
            top_k=top_k,
            language=language,
            min_year=year_min_selected,
            max_year=year_max_selected,
        )

        st.subheader("Results")
        # Show table
        st.dataframe(results, use_container_width=True)

        # Optional: nicer “cards”
        st.subheader("Book cards")
        for rank, row in results.iterrows():
            with st.container(border=True):
                st.markdown(f"### {rank}. {row['title']}")
                st.markdown(f"**Author:** {row['author']}  \n"
                            f"**Year:** {row['published_year']}  \n"
                            f"**Language:** {row['language']}  \n"
                            f"**Similarity:** {row['similarity']:.3f}")
                st.markdown(f"**Subjects:** {row['subjects']}")
                if isinstance(row.get("cover"), str) and row["cover"].startswith("http"):
                    st.markdown(f"**Cover link:** {row['cover']}")
