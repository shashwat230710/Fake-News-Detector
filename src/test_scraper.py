from src.scraper import get_article_text

if __name__ == "__main__":
    url = "https://www.bbc.com/news/world-60525350"  # sample news article
    text = get_article_text(url)
    print("✅ Extracted Article Text:\n")
    print(text[:1000])  # print first 1000 chars
