
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
st.write("Describe what you're looking for:")

# Load data
df = pd.read_excel("AI Toy Finder-Sample Sheet1.xlsx")

# Combine title and description
df["combined_tags"] = df["Product Title"].fillna('') + " " + df["Product Description"].fillna('')

# User input
query = st.text_input("", placeholder="e.g., toy for a 4 year old who loves to climb")

def get_top_matches(query, threshold=0.3):
    vectorizer = TfidfVectorizer().fit_transform([query] + df["combined_tags"].tolist())
    similarity = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()
    df["similarity"] = similarity
    results = df[df["similarity"] >= threshold].sort_values(by="similarity", ascending=False)
    return results

if query:
    matches = get_top_matches(query, threshold=0.3)

    if matches.empty:
        matches = get_top_matches(query, threshold=0.1)

    if matches.empty:
        st.warning("No relevant results found. Try a different query.")
    else:
        st.subheader("🔍 Top Matching Toys")
        for _, row in matches.head(3).iterrows():
            st.markdown(f"### {row['Product Title']}")
            st.markdown(f"_{row['Product Description']}_")
            if 'Image Link' in row and pd.notna(row['Image Link']):
                st.image(row['Image Link'], width=300)
            st.markdown("---")
