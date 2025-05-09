
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Password protection
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

st.title("🧠 Snooplay Toy Finder (Demo)")
query = st.text_input("Describe what you're looking for:")

# Load and clean data
df = pd.read_excel("AI Toy Finder-Sample Sheet1.xlsx")
df.columns = df.columns.str.strip()

required_cols = ["Product Title", "Product Description", "Age", "Skills", "Play Type", "Mood", "Learning Outcome"]
if not all(col in df.columns for col in required_cols):
    st.error("The uploaded sheet is missing required columns.")
    st.stop()

df["combined_tags"] = df[required_cols].fillna("").agg(" ".join, axis=1)

# Perform search
if query:
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["combined_tags"])
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    df["similarity"] = similarities
    results = df[df["similarity"] > 0.3].sort_values(by="similarity", ascending=False).head(3)

    if results.empty:
        st.warning("No relevant results found. Try a different query.")
    else:
        for _, row in results.iterrows():
            st.markdown(f"### {row['Product Title']}")
            st.markdown(f"*{row['Product Description']}*")
            st.markdown(f"**Age:** {row['Age']}")
            st.markdown(f"**Skills:** {row['Skills']}")
            st.markdown(f"**Play Type:** {row['Play Type']}")
            st.markdown(f"**Mood:** {row['Mood']}")
            st.markdown(f"**Learning Outcome:** {row['Learning Outcome']}")
            if pd.notna(row.get("Image URL")):
                st.image(row["Image URL"], use_column_width=True)
            st.markdown("---")
