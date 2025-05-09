
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
st.title("🧠 Snooplay Toy Finder (Character N-Gram Match)")
st.write("### Describe what you're looking for:")

# Load and prepare data
df = pd.read_excel("AI Toy Finder-Sample Sheet1.xlsx")
df = df.fillna("")
df["combined_text"] = df["Product Title"].astype(str) + " " + df["Product Description"].astype(str)

# Input from user
query = st.text_input("")

# Character-level TF-IDF vectorizer (4-6 char n-grams)
if query:
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(4, 6))
    tfidf_matrix = vectorizer.fit_transform(df["combined_text"].tolist() + [query])
    query_vector = tfidf_matrix[-1]
    doc_vectors = tfidf_matrix[:-1]

    similarities = cosine_similarity(query_vector, doc_vectors).flatten()
    df["similarity"] = similarities
    top_matches = df.sort_values(by="similarity", ascending=False).head(3)

    if top_matches.empty or top_matches["similarity"].max() == 0:
        st.warning("No relevant results found. Try a different query.")
    else:
        st.subheader("🔍 Top Matching Toys")
        if top_matches["similarity"].max() < 0.2:
            st.info("These results may not be exact matches, but are the closest we could find:")

        for _, row in top_matches.iterrows():
            st.markdown(f"### {row['Product Title']}")
            st.markdown(f"_Similarity Score: {row['similarity']*100:.1f}%_")
            short_desc = row["Product Description"][:200] + ("..." if len(row["Product Description"]) > 200 else "")
            st.markdown(f"*{short_desc}*")
            if "Image URL" in row and isinstance(row["Image URL"], str) and row["Image URL"].startswith("http"):
                st.image(row["Image URL"], width=300)
            st.markdown("---")
