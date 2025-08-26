import streamlit as st
from ..predict import analyze

st.set_page_config(page_title="Fake News Detector + Summarizer", page_icon="📰", layout="centered")

st.title("📰 Fake News Detector + Summarizer")
st.write("Paste a news **URL** or **Text** below and click Analyze.")

tab1, tab2 = st.tabs(["URL", "Raw Text"])

with tab1:
    url = st.text_input("Article URL", placeholder="https://example.com/news-article")
    if st.button("Analyze URL", use_container_width=True):
        with st.spinner("Analyzing..."):
            result = analyze(url=url, do_summary=True)
        st.write(result)

with tab2:
    text = st.text_area("Article Text", height=250, placeholder="Paste the article text here...")
    if st.button("Analyze Text", use_container_width=True):
        with st.spinner("Analyzing..."):
            result = analyze(text=text, do_summary=True)
        st.write(result)

st.caption("Baseline: TF‑IDF + Logistic Regression. Summarizer: distilBART (Hugging Face).")