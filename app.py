import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------- Password Protection ----------------
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    password = st.text_input("Enter password:", type="password")
    if password == "snoop321":
        st.session_state["authenticated"] = True
        st.experimental_rerun()
    elif password:
        st.error("Incorrect password")
    st.stop()

# ---------------- Load Data ----------------
df = pd.read_excel("AI Toy Finder-Sample Sheet1.xlsx")
df.dropna(subset=["Product Title", "Product Description", "GPT Response (Raw Text)"], inplace=True)

# ---------------- Vectorization ----------------
tfidf = TfidfVectorizer(stop_words='english')
corpus = df["Product Title"].astype(str) + " " + df["Product Description"].astype(str) + " " + df["GPT Response (Raw Text)"].astype(str)
tfidf_matrix = tfidf.fit_transform(corpus)

# ---------------- UI ----------------
st.title("🧠 Snooplay Toy Finder (Demo)")
query = st.text_input("Describe what you're looking for:")

if query:
    query_vec = tfidf.transform([query])
    sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    df["score"] = sim_scores
    top_matches = df.sort_values("score", ascending=False).head(5)

    st.subheader("🔍 Top Matching Toys")
    for _, row in top_matches.iterrows():
        st.markdown(f"### {row['Product Title']}")
        st.markdown(f"*{row['Product Description']}*")
        st.markdown(f"**Skills:** {row['Skills']}  ")
        st.markdown(f"**Play Type:** {row['Play Type']}  ")
        st.markdown(f"**Mood:** {row['Mood']}  ")
        st.markdown(f"**Learning Outcome:** {row['Learning']}  ")
        if pd.notna(row['Image URL']):
            st.image(row['Image URL'], use_column_width=True)
        st.markdown("---")
