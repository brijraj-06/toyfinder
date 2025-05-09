import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# --------------- Password Protection ---------------
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
    else:
        st.stop()

# ---------------- Toy Finder UI ----------------
st.title("🧠 Snooplay Toy Finder (Demo)")
query = st.text_input("Describe what you're looking for:")

# Load Excel data
df = pd.read_excel("AI Toy Finder-Sample Sheet1.xlsx")

# Combine title, description, and AI tags into one searchable field
df["combined_tags"] = df[["Product Title", "Product Description", "Age", "Skills", "Play Type", "Mood", "Learning"]].fillna("").agg(" ".join, axis=1)

# Run only if a query is entered
if query:
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(df["combined_tags"].tolist() + [query])
    similarities = cosine_similarity(vectors[-1], vectors[:-1]).flatten()

    # Get top 3 matches
    top_indices = similarities.argsort()[::-1][:3]
    top_scores = similarities[top_indices]

    if top_scores.max() < 0.2:
        st.warning("No relevant results found. Try a different query.")
    else:
        st.subheader("🔍 Top Matching Toys")
        for idx in top_indices:
            if top_scores[idx] < 0.2:
                continue
            row = df.iloc[idx]
            st.markdown(f"### {row['Product Title']}")
            st.image(row["Image URL"], use_column_width=True)
            st.markdown(f"*{row['Product Description']}*")
            st.markdown(f"**Age:** {row['Age']}")
            st.markdown(f"**Skills:** {row['Skills']}")
            st.markdown(f"**Play Type:** {row['Play Type']}")
            st.markdown(f"**Mood:** {row['Mood']}")
            st.markdown(f"**Learning Outcome:** {row['Learning']}")
            st.markdown("---")
