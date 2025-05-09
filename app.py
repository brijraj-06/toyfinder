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

# ---------------- Load Data ----------------
df = pd.read_excel("AI Toy Finder-Sample Sheet1.xlsx")

# Combine title and description for matching
df["combined_text"] = df[["Product Title", "Product Description"]].fillna("").agg(" ".join, axis=1)

# ---------------- Streamlit UI ----------------
st.markdown("## 🧠 Snooplay Toy Finder (Demo)")
user_query = st.text_input("Describe what you're looking for:")

if user_query:
    # Vectorize input and data
    vectorizer = TfidfVectorizer().fit(df["combined_text"].tolist() + [user_query])
    query_vec = vectorizer.transform([user_query])
    data_vecs = vectorizer.transform(df["combined_text"])

    # Compute similarity
    similarities = cosine_similarity(query_vec, data_vecs).flatten()

    # Get top 3 results
    df["similarity"] = similarities
    top_matches = df.sort_values(by="similarity", ascending=False).head(3)

    if top_matches["similarity"].max() < 0.1:
        st.warning("No relevant results found. Try a different query.")
    else:
        st.markdown("### 🔍 Top Matching Toys")
        for _, row in top_matches.iterrows():
            st.markdown(f"**{row['Product Title']}**")
            st.markdown(f"*{row['Product Description']}*")
            st.markdown(f"**Skills:** {row['Skills']}")
            st.markdown(f"**Play Type:** {row['Play Type']}")
            st.markdown(f"**Mood:** {row['Mood']}")
            st.markdown(f"**Learning Outcome:** {row['Learning']}")
            st.markdown(f"**Age Group:** {row['Age']}")
            if pd.notna(row.get("Image Link")):
                st.image(row["Image Link"], width=300)
            st.markdown(f"**Similarity Score:** {round(row['similarity'] * 100, 2)}%")
            st.markdown("---")
