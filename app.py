
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ Password Protection ------------------
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

# ------------------ App Title ------------------
st.markdown("<h1 style='font-size: 42px;'>🧠 Snooplay Toy Finder (Demo)</h1>", unsafe_allow_html=True)
st.write("Describe what you're looking for:")

# ------------------ Load Data ------------------
df = pd.read_excel("AI Toy Finder-Sample Sheet1.xlsx")

# Ensure relevant columns exist
df = df.dropna(subset=["Product Title", "Product Description"])

# Combine just title and description
df["combined_text"] = df["Product Title"].astype(str).str.lower() + " " + df["Product Description"].astype(str).str.lower()

# ------------------ User Input ------------------
query = st.text_input("", placeholder="e.g., toy for a 5 year old who loves to build")

if query:
    query = query.lower()

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["combined_text"].tolist() + [query])
    
    similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    
    df["similarity"] = similarities
    filtered_df = df[df["similarity"] >= 0.3]

    if filtered_df.empty:
        filtered_df = df[df["similarity"] >= 0.1]

    if filtered_df.empty:
        st.warning("No relevant results found. Try a different query.")
    else:
        top_matches = filtered_df.sort_values(by="similarity", ascending=False).head(3)
        st.subheader("🔍 Top Matching Toys")
        for _, row in top_matches.iterrows():
            st.markdown(f"### {row['Product Title']}")
            st.markdown(f"*{row['Product Description']}*")
            if "Image Link" in row and pd.notna(row["Image Link"]):
                st.image(row["Image Link"], use_column_width=True)
