import streamlit as st
import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# norm() helper function
def norm(s: str) -> str:
    s = str(s).lower()
    return re.sub(r"[^a-z0-9]+", "", s)  # remove spaces/punctuation

def clean_for_tfidf(s: str) -> str:
    s = str(s)
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)  # tonyChapman -> tony Chapman
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# import
st.set_page_config(page_title="Bookommendor", layout="centered")


# ----------------------------
# Session state defaults
# ----------------------------
if "page" not in st.session_state:
    st.session_state.page = "main"
if "query" not in st.session_state:
    st.session_state.query = ""

# ----------------------------
# Cached loaders
# ----------------------------
@st.cache_data
def load_books():
    return pd.read_csv("books_with_content.csv")

@st.cache_data
def load_embeddings():
    return np.load("book_embeddings.npy")

@st.cache_resource
def load_sbert_model():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

@st.cache_resource
def build_tfidf_matrix(_df: pd.DataFrame):
    title = _df["title"].fillna("").astype(str) if "title" in _df.columns else ""
    author = _df["author"].fillna("").astype(str) if "author" in _df.columns else ""
    text = (title + " " + author).map(clean_for_tfidf)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1
    )
    tfidf_matrix = vectorizer.fit_transform(text)
    return vectorizer, tfidf_matrix

# ----------------------------
# Load data
# ----------------------------
books_df = load_books()
embeddings = load_embeddings()
emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
tfidf_vectorizer, tfidf_matrix = build_tfidf_matrix(books_df)

# ----------------------------
# Recommendation functions
# ----------------------------
def recommend_sbert(query: str, top_k: int = 10):
    model = load_sbert_model()
    query_vec = model.encode([query], normalize_embeddings=True)  # (1, 384)

    scores = (emb_norm @ query_vec.T).ravel()  # (N,)
    top_idx = np.argsort(scores)[::-1][:top_k]

    results = books_df.iloc[top_idx].copy()
    results["sbert_score"] = scores[top_idx]  
    return results


def recommend_sbert_from_seed(seed_idx: int, top_k: int = 50):
    seed_vec = emb_norm[seed_idx].reshape(1, -1)
    scores = (emb_norm @ seed_vec.T).ravel()
    top_idx = np.argsort(scores)[::-1]

    top_idx = [i for i in top_idx if i != seed_idx][:top_k]

    results = books_df.iloc[top_idx].copy()
    results["sbert_score"] = scores[top_idx]  # ✅ ADD SCORES
    return results


def recommend_tfidf_all_matches(query: str, threshold: float = 0.20, max_matches: int = 100):
    """
    Returns ALL TF-IDF matches whose similarity >= threshold.
    max_matches is just a safety cap (in case query matches thousands).
    """
    q = clean_for_tfidf(query)
    q_vec = tfidf_vectorizer.transform([q])
    sims = cosine_similarity(q_vec, tfidf_matrix).ravel()

    # keep ONLY strong matches
    idx = np.where(sims >= threshold)[0]

    # sort by similarity (highest first)
    idx = idx[np.argsort(sims[idx])[::-1]]

    # safety cap
    idx = idx[:max_matches]

    results = books_df.iloc[idx].copy()
    results["tfidf_score"] = sims[idx]
    return results

def has_strong_tfidf_match(tfidf_results: pd.DataFrame, threshold: float = 0.20) -> bool:
    if tfidf_results is None or tfidf_results.empty:
        return False
    best = float(tfidf_results["tfidf_score"].max())
    return best >= threshold

def has_good_sbert_match(sbert_df: pd.DataFrame, min_score: float = 0.45) -> bool:
    if sbert_df is None or sbert_df.empty or "sbert_score" not in sbert_df.columns:
        return False
    return float(sbert_df["sbert_score"].max()) >= min_score

# ----------------------------
# UI helpers
# ----------------------------
def render_book_cards(df, max_items=10):
    """
    Renders a list of books as nice cards with cover image + metadata.
    """
    cover_col = "cover" if "cover" in df.columns else None
    year_col = "published_year" if "published_year" in df.columns else ("year" if "year" in df.columns else None)

    for _, row in df.head(max_items).iterrows():
        col_img, col_text = st.columns([1, 5], vertical_alignment="top")

        # --- image ---
        with col_img:
            cover_val = row.get(cover_col) if cover_col else None
            cover_str = "" if pd.isna(cover_val) else str(cover_val).strip()

            if cover_str in ["", "0", "nan", "None"]:
                cover_str = ""

            if cover_str.startswith("http"):
                st.image(cover_str, width=110)
            else:
                st.image("https://via.placeholder.com/110x160?text=No+Cover", width=110)

        # --- text ---
        with col_text:
            title = str(row.get("title", "Untitled"))
            author = str(row.get("author", "Unknown author"))
            lang = str(row.get("language", "unknown"))

            year_txt = ""
            if year_col and pd.notna(row.get(year_col)):
                year_txt = f"{int(row[year_col])}" if str(row[year_col]).isdigit() else str(row[year_col])

            meta_parts = []
            if author and author != "nan":
                meta_parts.append(f"**Author:** {author}")
            if year_txt:
                meta_parts.append(f"**Year:** {year_txt}")
            if lang and lang != "nan":
                meta_parts.append(f"**Language:** {lang}")

            st.markdown(f"### {title}")
            st.markdown(" • ".join(meta_parts))

        st.divider()

# ----------------------------
# Pages
# ----------------------------
def main_page():
    st.markdown("<br><br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.image("pexels-suzyhazelwood-1333742.jpg", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        "<h1 style='text-align:center; font-size:60px; font-weight:800; margin:0;'>Bookommendor</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p style='text-align:center; font-size:18px; color:#6b7280;'>Find books by title, author, or topic</p>",
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        with st.form("search_form", clear_on_submit=False):
            query = st.text_input(
                label="",
                placeholder="Search: e.g. Harari, Atomic Habits...",
                value=st.session_state.query,
                key="query_input",
            )
            submitted = st.form_submit_button("Search", use_container_width=True)

    if submitted:
        st.session_state.query = query
        st.session_state.page = "results"
        st.rerun()


def make_key(df: pd.DataFrame) -> pd.Series:
    # Unique key = title + author (works well for your dataset)
    t = df["title"].fillna("").astype(str).str.lower().str.strip() if "title" in df.columns else ""
    a = df["author"].fillna("").astype(str).str.lower().str.strip() if "author" in df.columns else ""
    return (t + "||" + a)


def results_page():
    # ✅ override the global centered CSS ONLY for page 2
    st.markdown(
        """
        <style>
          .block-container{
            max-width: 1200px !important;   /* or 100% */
            margin-left: 2rem !important;   /* left aligned */
            margin-right: auto !important;
            text-align: left !important;
          }

          /* undo centering rules you set globally */
          [data-testid="stImage"] { justify-content: flex-start !important; }
          [data-testid="stTextInput"] { justify-content: flex-start !important; }
          [data-testid="stButton"] { justify-content: flex-start !important; }

          [data-testid="stButton"] button { width: auto !important; }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Results")

    if st.button("⬅ Back"):
        st.session_state.page = "main"
        st.rerun()

    query = st.session_state.query.strip()
    st.write(f"Showing results for: **{query}**")

    if len(query) < 3:
        st.info("Please enter a longer query.")
        return

    # Fixed values
    threshold = 0.20
    similar_k = 5
    min_sbert_score = 0.40  # ✅ ADD (tune 0.40–0.50)
    # ----------------------------
    # 1) TF-IDF: return ALL exact matches above threshold
    # ----------------------------
    with st.spinner("Checking for exact matches..."):
        tfidf_results = recommend_tfidf_all_matches(query, threshold=threshold)

    strong_match = not tfidf_results.empty

    # ----------------------------
    # 2) SBERT: if we have an exact match -> seed-based recommendations
    # ----------------------------
    with st.spinner("Finding semantic recommendations..."):

        if strong_match:
            # use best exact match as seed
            seed_row = tfidf_results.iloc[0]
            seed_title = seed_row["title"]
            seed_author = seed_row["author"]

            # locate seed index in books_df
            all_keys = (books_df["title"].fillna("").map(norm) + "||" +
                        books_df["author"].fillna("").map(norm))
            seed_key = norm(seed_title) + "||" + norm(seed_author)

            seed_matches = np.where(all_keys.values == seed_key)[0]

            if len(seed_matches) > 0:
                seed_idx = int(seed_matches[0])
                sbert_candidates = recommend_sbert_from_seed(seed_idx, top_k=50)
            else:
                # fallback to normal semantic search
                sbert_candidates = recommend_sbert(query, top_k=50)

        else:
            # no exact match -> normal semantic search
            sbert_candidates = recommend_sbert(query, top_k=50)

    # remove duplicates from exact results
    if strong_match:
        exact_keys = set(make_key(tfidf_results))
        sbert_candidates = sbert_candidates[~make_key(sbert_candidates).isin(exact_keys)]

    sbert_results = sbert_candidates.head(similar_k)

    good_semantic = has_good_sbert_match(sbert_candidates.head(20), min_score=min_sbert_score)

    if not strong_match and not good_semantic:
        st.warning(
            "No strong semantic match found for your query. "
            "Try different keywords (or the dataset may not contain this topic)."
        )

    # ----------------------------
    # 3) Display
    # ----------------------------
    if strong_match:
        st.subheader(f"Top results (Exact match) — {len(tfidf_results)} found")
        tfidf_results = tfidf_results.drop(columns=["tfidf_score"], errors="ignore")
        render_book_cards(tfidf_results, max_items=len(tfidf_results))  # show ALL exact matches

        st.subheader(f"Recommended books (Similar meaning) — Top {similar_k}")
        display_df = sbert_results.drop(columns=["sbert_score"], errors="ignore")
        render_book_cards(display_df, max_items=similar_k)

    else:
        st.caption("No strong title/author match found — showing semantic recommendations.")
        st.subheader(f"Recommended books (Semantic Search) — Top {similar_k}")
        display_df = sbert_results.drop(columns=["sbert_score"], errors="ignore")
        render_book_cards(display_df, max_items=similar_k)


# ----------------------------
# Router
# ----------------------------
if st.session_state.page == "main":
    main_page()
else:
    results_page()
