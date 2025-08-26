import streamlit as st
from src.scraper import get_article_text
from src.predict import analyze

# Streamlit app title
st.title("📰 Fake News Detector & Summarizer")

# Input URL
url = st.text_input("Enter a news article URL:")

if st.button("Analyze"):
    if url:
        with st.spinner("Fetching and analyzing article..."):
            # Extract article
            article_text = get_article_text(url)

            if article_text:
                # Run prediction
                result = analyze(article_text)

                # Show results
                st.subheader("🔍 Prediction")
                st.write(f"**Label:** {result['label']}")
                st.write(f"**Probability Fake:** {result['probability_fake']:.2f}")

                st.subheader("📝 Summary")
                st.write(result['summary'])

                st.subheader("📖 Full Extracted Article")
                st.text_area("Article Text", article_text, height=300)

            else:
                st.error("❌ Could not extract article text. Please check URL.")
    else:
        st.warning("Please enter a URL.")
