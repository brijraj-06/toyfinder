
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
        st.rerun()
    elif password:
        st.error("Incorrect password")
        st.stop()

# ---------------- App Logic ----------------
st.markdown("## 🧠 Snooplay Toy Finder (Demo)")
query = st.text_input("Describe what you're looking for:")

# Load the cleaned data file
df = pd.read_excel("AI Toy Finder-Sample Sheet1.xlsx")

# Combine relevant columns for search
df["combined_tags"] = df[["Product Title", "Product Description", "Age", "Skills", "Play Type", "Mood", "Learning Outcome"]].fillna("").agg(" ".join, axis=1)

# Only run matching if query is given
if query:
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([query] + df["combined_tags"].tolist())
    similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    df["score"] = similarities

    top_matches = df[df["score"] > 0.3].sort_values(by="score", ascending=False).head(3)

    if not top_matches.empty:
        for _, row in top_matches.iterrows():
            st.markdown(f"### {row['Product Title']}")
            st.markdown(f"*{row['Product Description']}*")
            st.markdown(f"**Age Group:** {row['Age']}")
            st.markdown(f"**Skills:** {row['Skills']}")
            st.markdown(f"**Play Type:** {row['Play Type']}")
            st.markdown(f"**Mood:** {row['Mood']}")
            st.markdown(f"**Learning Outcome:** {row['Learning Outcome']}")
            if pd.notna(row['Image URL']):
                st.image(row['Image URL'], width=300)
            st.markdown("---")
    else:
        st.warning("No relevant results found. Try a different query.")
