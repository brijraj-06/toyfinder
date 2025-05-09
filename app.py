
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

# ---------------- Toy Finder Logic ----------------
st.title("🧠 Snooplay Toy Finder (Demo)")

query = st.text_input("Describe what you're looking for:")

# Load the dataset
df = pd.read_excel("AI Toy Finder-Sample Sheet1.xlsx")

# Combine relevant tags into one text column
df["combined_tags"] = df[[
    "Age (from GPT)",
    "Skills (from GPT)",
    "Play Type (from GPT)",
    "Mood (from GPT)",
    "Learning Outcome (from GPT)"
]].fillna("").agg(" ".join, axis=1)

# Process the query and match top results
if query:
    all_text = df["combined_tags"].tolist() + [query]
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(all_text)
    similarities = cosine_similarity(vectors[-1], vectors[:-1]).flatten()

    df["similarity"] = similarities
    filtered = df[df["similarity"] > 0.3].sort_values("similarity", ascending=False).head(3)

    if not filtered.empty:
        st.markdown("🔍 **Top Matching Toys**")
        for _, row in filtered.iterrows():
            st.markdown(f"### {row['Product Title']}")
            st.markdown(f"_{row['Product Description']}_")
            st.markdown(f"![Image]({row['Image URL']})")
            st.markdown(f"**Skills:** {row['Skills (from GPT)']}")
            st.markdown(f"**Play Type:** {row['Play Type (from GPT)']}")
            st.markdown(f"**Mood:** {row['Mood (from GPT)']}")
            st.markdown(f"**Learning Outcome:** {row['Learning Outcome (from GPT)']}")
            st.markdown("---")
    else:
        st.warning("No relevant results found. Try a different query.")
