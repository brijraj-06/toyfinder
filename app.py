
import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------- Password Protection --------------
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

# -------------- Toy Finder Logic --------------
st.title("🧠 Snooplay Toy Finder (Demo)")

query = st.text_input("Describe what you're looking for:")

if query:
    df = pd.read_excel("AI Toy Finder-Sample Sheet1.xlsx")
    df["combined_text"] = df["Product Title"].fillna("") + " " + df["Product Description"].fillna("") + " " + df["Skills"].fillna("") + " " + df["Play Type"].fillna("") + " " + df["Mood"].fillna("") + " " + df["Learning"].fillna("")

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["combined_text"])
    query_vec = vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    df["score"] = similarity_scores
    results = df.sort_values(by="score", ascending=False).head(5)

    st.subheader("🔍 Top Matching Toys")
    for _, row in results.iterrows():
        st.markdown(f"### {row['Product Title']}")
        st.image(row["Image URL"], width=200)
        st.markdown(f"_{row['Product Description']}_")
        st.markdown(f"**Skills:** {row['Skills']}")
        st.markdown(f"**Play Type:** {row['Play Type']}")
        st.markdown(f"**Mood:** {row['Mood']}")
        st.markdown(f"**Learning Outcome:** {row['Learning']}")
        st.markdown("---")
