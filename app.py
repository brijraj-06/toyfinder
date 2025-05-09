
import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------- Password Protection ----------------
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    password = st.text_input("Enter password to access the Toy Finder:", type="password")
    if password == "snoop321":
        st.session_state["authenticated"] = True
        st.rerun()
    elif password:
        st.error("Incorrect password")
        st.stop()

# ---------------- App UI ----------------
st.title("🧠 Snooplay Toy Finder (Demo)")
query = st.text_input("Describe what you're looking for:")

# ---------------- Load and Process Data ----------------
@st.cache_data
def load_data():
    df = pd.read_excel("AI Toy Finder-Sample Sheet1.xlsx")
    df = df.dropna(subset=["Product Title", "Product Description"])
    df["combined_text"] = df["Product Title"].astype(str) + " " + df["Product Description"].astype(str)
    return df

df = load_data()

# ---------------- Vectorization Function ----------------
def get_top_matches(query, df, threshold=0.3):
    tfidf = TfidfVectorizer(stop_words="english")
    vectors = tfidf.fit_transform(df["combined_text"].tolist() + [query])
    cosine_sim = cosine_similarity(vectors[-1], vectors[:-1]).flatten()
    df["similarity"] = cosine_sim
    filtered = df[df["similarity"] >= threshold].sort_values(by="similarity", ascending=False).head(3)
    return filtered

# ---------------- Matching Logic ----------------
if query:
    matches = get_top_matches(query, df, threshold=0.3)

    if matches.empty:
        matches = get_top_matches(query, df, threshold=0.1)

    if matches.empty:
        st.warning("No relevant results found. Try a different query.")
    else:
        st.subheader("🔍 Top Matching Toys")
        for _, row in matches.iterrows():
            st.markdown(f"### {row['Product Title']}")
            st.markdown(f"*{row['Product Description']}*")
            st.markdown(f"**Image:**")
            st.image(row["Image URL"], width=300)
            st.markdown("---")
