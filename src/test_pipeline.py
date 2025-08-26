from src.scraper import get_article_text
from src.preprocess import clean_text
from src.predict import analyze
from src.summarizer import summarize_text

if __name__ == "__main__":
    url = "https://www.bbc.com/news/world-60525350"

    # Step 1: Scrape
    raw_text = get_article_text(url)
    print("✅ Scraped article length:", len(raw_text))

    # Step 2: Preprocess
    cleaned = clean_text(raw_text)
    print("\n✅ Cleaned text (first 300 chars):\n", cleaned[:300])

    # Step 3: Fake news detection
    label = analyze(raw_text)
    print("\n🔍 Prediction:", label)

    # Step 4: Summarization
    summary = summarize_text(raw_text)
    print("\n📝 Summary:\n", summary)
