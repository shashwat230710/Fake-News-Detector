import streamlit as st
from src.scraper import get_article_text
from src.predict import analyze

# Streamlit app title
st.title("📰 Fake News Detector & Summarizer")

# Option to choose input type
input_type = st.radio("Choose input type:", ("URL", "Direct Text"))

news_text = ""

if input_type == "URL":
    url = st.text_input("Enter a news article URL:")
    if st.button("Analyze URL"):
        if url.strip():
            with st.spinner("Fetching and analyzing article..."):
                # Extract article
                news_text = get_article_text(url)

                if news_text:
                    result = analyze(news_text)

                    # Show results safely
                    st.subheader("🔍 Prediction")
                    st.write(f"**Label:** {result['label']}")
                    st.write(f"**Probability Fake:** {result['probability_fake']:.2f}")

                    # Show summary only if it exists
                    if 'summary' in result:
                        st.subheader("📝 Summary")
                        st.write(result['summary'])
                    else:
                        st.info("📝 No summary available for this article.")

                    st.subheader("📖 Full Extracted Article")
                    st.text_area("Article Text", news_text, height=300)
                else:
                    st.error("❌ Could not extract article text. Please check URL.")
        else:
            st.warning("Please enter a URL.")

elif input_type == "Direct Text":
    news_text = st.text_area("Enter the news text here:", height=200)
    if st.button("Analyze Text"):
        if news_text.strip():
            with st.spinner("Analyzing text..."):
                result = analyze(news_text)

                # Show results safely
                st.subheader("🔍 Prediction")
                st.write(f"**Label:** {result['label']}")
                st.write(f"**Probability Fake:** {result['probability_fake']:.2f}")

                # Show summary only if it exists
                if 'summary' in result:
                    st.subheader("📝 Summary")
                    st.write(result['summary'])
                else:
                    st.info("📝 No summary available for this text.")

                st.subheader("📖 Full Text")
                st.text_area("News Text", news_text, height=300)
        else:
            st.warning("Please enter some text.")
