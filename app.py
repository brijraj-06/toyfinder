
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

# ---------------- Toy Finder App ----------------
st.title("🧠 Snooplay Toy Finder (Demo)")
query = st.text_input("Describe what you're looking for:")

# Load data
file_path = "AI Toy Finder-Sample Sheet1.xlsx"
try:
    df = pd.read_excel(file_path)
except FileNotFoundError:
    st.error("Product data file not found. Please upload the correct Excel file.")
    st.stop()

# Combine tags for similarity
df["combined_tags"] = df[["Age", "Skills", "Play Type", "Mood", "Learning Outcome"]].fillna("").agg(" ".join, axis=1)

# TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["combined_tags"])

# Process query
if query:
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    df["similarity"] = similarities

    # Sort and filter
    top_matches = df.sort_values(by="similarity", ascending=False).head(3)

    st.markdown("🔍 **Top Matching Toys**")
    for _, row in top_matches.iterrows():
        st.markdown(f"### {row['Product Title']}")
        st.markdown(f"*{row['Product Description']}*")
        st.markdown(f"**Skills:** {row.get('Skills', 'N/A')}")
        st.markdown(f"**Play Type:** {row.get('Play Type', 'N/A')}")
        st.markdown(f"**Mood:** {row.get('Mood', 'N/A')}")
        st.markdown(f"**Learning Outcome:** {row.get('Learning Outcome', 'N/A')}")
        if pd.notna(row.get("Image URL", None)):
            st.image(row["Image URL"], width=300)
        st.markdown("---")
