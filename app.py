
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
    else:
        st.stop()

# Load Excel file
df = pd.read_excel("AI Toy Finder-Sample Sheet1.xlsx")

# Combine relevant columns into a single text field for similarity search
df["combined_tags"] = df[["Product Title", "Product Description", "Age", "Skills", "Play Type", "Mood", "Learning Outcome"]].fillna("").agg(" ".join, axis=1)

# User input
st.title("🧠 Snooplay Toy Finder (Demo)")
query = st.text_input("Describe what you're looking for:")

if query:
    # Compute similarity
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([query] + df["combined_tags"].tolist())
    cosine_sim = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

    # Threshold and top results
    df["similarity"] = cosine_sim
    top_matches = df[df["similarity"] > 0.1].sort_values(by="similarity", ascending=False).head(3)

    if top_matches.empty:
        st.warning("No relevant results found. Try a different query.")
    else:
        st.subheader("🔍 Top Matching Toys")
        for _, row in top_matches.iterrows():
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
