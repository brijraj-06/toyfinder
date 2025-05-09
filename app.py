
import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Simple password protection ---
def check_password():
    def password_entered():
        if st.session_state["password"] == "snoop321":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Enter password to access the Toy Finder:", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Enter password to access the Toy Finder:", type="password", on_change=password_entered, key="password")
        st.error("Incorrect password")
        return False
    else:
        return True

if not check_password():
    st.stop()

# --- Main Toy Finder App ---
st.title("🧠 Snooplay Toy Finder (Demo)")

df = pd.read_excel("AI Toy Finder-Sample Sheet1.xlsx")

def combine_tags(row):
    return f"{row['Product Title']} {row['Product Description']} {row['Skills (from GPT)']} {row['Play Type (from GPT)']} {row['Mood (from GPT)']} {row['Learning Outcome (from GPT)']}"

df['combined_text'] = df.apply(combine_tags, axis=1)

query = st.text_input("Describe what you're looking for:", placeholder="e.g. Gift for 5-year-old who loves exploring")

if query:
    corpus = df['combined_text'].tolist() + [query]
    vectorizer = TfidfVectorizer().fit_transform(corpus)
    cosine_sim = cosine_similarity(vectorizer[-1], vectorizer[:-1])
    scores = cosine_sim.flatten()

    top_indices = scores.argsort()[::-1][:5]

    st.subheader("🔍 Top Matching Toys")
    for i in top_indices:
        st.image(df.iloc[i]['Image URL'], width=250)
        st.markdown(f"**{df.iloc[i]['Product Title']}**")
        st.markdown(f"_{df.iloc[i]['Product Description']}_")
        st.markdown(f"**Skills:** {df.iloc[i]['Skills (from GPT)']}")
        st.markdown(f"**Play Type:** {df.iloc[i]['Play Type (from GPT)']}")
        st.markdown(f"**Mood:** {df.iloc[i]['Mood (from GPT)']}")
        st.markdown(f"**Learning Outcome:** {df.iloc[i]['Learning Outcome (from GPT)']}")
        st.markdown("---")
