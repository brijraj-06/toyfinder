
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

# ---------------- App Title ----------------
st.markdown("## 🧠 Snooplay Toy Finder (Demo)")
st.write("Describe what you're looking for:")

# ---------------- Load Data ----------------
df = pd.read_excel("AI Toy Finder-Sample Sheet1.xlsx")

# Combine only Product Title and Description for semantic search
df["combined_text"] = df["Product Title"].astype(str) + " " + df["Product Description"].astype(str)

# ---------------- Input Query ----------------
query = st.text_input("")

if query:
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(df["combined_text"].tolist() + [query])
    query_vec = vectors[-1]
    product_vecs = vectors[:-1]

    # Compute Similarities
    similarities = cosine_similarity(query_vec, product_vecs).flatten()
    df["similarity"] = similarities

    # First, try with threshold 0.3
    top_matches = df[df["similarity"] > 0.3].sort_values(by="similarity", ascending=False).head(3)

    # If no results, lower the threshold to 0.1
    if top_matches.empty:
        top_matches = df[df["similarity"] > 0.1].sort_values(by="similarity", ascending=False).head(3)

    # Show results or fallback message
    if not top_matches.empty:
        st.markdown("### 🔍 Top Matching Toys")
        for _, row in top_matches.iterrows():
            st.subheader(row["Product Title"])
            st.markdown(f"*{row['Product Description']}*")
            st.markdown(f"**Skills:** {row['Skills']}")
            st.markdown(f"**Play Type:** {row['Play Type']}")
            st.markdown(f"**Mood:** {row['Mood']}")
            st.markdown(f"**Learning Outcome:** {row['Learning']}")
            if pd.notna(row.get("Image URL", None)):
                st.image(row["Image URL"], width=300)
            st.markdown("---")
    else:
        st.warning("No relevant results found. Try a different query.")
