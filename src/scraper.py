import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import re

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36'
}

def get_article_text(url: str) -> str:
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
    except Exception as e:
        return f""
    soup = BeautifulSoup(r.text, 'lxml')

    # Prefer content within <article>, fallback to all <p>
    article = soup.find('article')
    paragraphs = []
    if article:
        paragraphs = [p.get_text(" ", strip=True) for p in article.find_all('p')]
    if not paragraphs:
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all('p')]

    text = " ".join(paragraphs)
    # Basic cleanup
    text = re.sub(r"\s+", " ", text).strip()
    return text