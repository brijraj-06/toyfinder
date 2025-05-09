
import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------- Password Protection --------------
if "authenticated" not in st.session_state:
    st.session_state.update({"authenticated": False})

if not st.session_state["authenticated"]:
    password = st.text_input("Enter password:", type="password")
    if password == "snoop321":
        st.session_state.update({"authenticated": True})
        st.rerun()
    elif password:
        st.error("Incorrect password")
        st.stop()

# -------------- Toy Finder App --------------
st.title("🧠 Snooplay Toy Finder (Demo)")

df = pd.read_excel("AI Toy Finder-Sample Sheet1.xlsx")

user_input = st.text_input("Describe what you're looking for:")

if user_input:
    corpus = df["Product Description"].astype(str).tolist() + [user_input]
    vectorizer = TfidfVectorizer(stop_words="english").fit(corpus)
    embeddings = vectorizer.transform(corpus)

    query_vec = embeddings[-1]
    toy_vecs = embeddings[:-1]

    similarities = cosine_similarity(query_vec, toy_vecs).flatten()
    df["Score"] = similarities
    results = df.sort_values(by="Score", ascending=False).head(5)

    st.subheader("🔍 Top Matching Toys")
    for _, row in results.iterrows():
        st.markdown(f"### {row['Product Title']}")
        if pd.notna(row.get("Image URL")):
            st.image(row["Image URL"], width=200)
        st.markdown(f"_{row['Product Description']}_")
        st.markdown(f"**Skills:** {row['Skills']}")
        st.markdown(f"**Play Type:** {row['Play Type']}")
        st.markdown(f"**Mood:** {row['Mood']}")
        st.markdown(f"**Learning Outcome:** {row['Learning']}")
        st.markdown("---")
