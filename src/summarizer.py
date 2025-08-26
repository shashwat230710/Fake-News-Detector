from typing import List
from transformers import pipeline

# Lazy global
_summarizer = None

def _get_summarizer():
    global _summarizer
    if _summarizer is None:
        # You can switch to 'facebook/bart-large-cnn' for higher quality
        _summarizer = pipeline('summarization', model='sshleifer/distilbart-cnn-12-6')
    return _summarizer

def _chunk_text(text: str, max_words: int = 800) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i:i+max_words]))
    return chunks

def summarize_text(text: str, max_words_chunk: int = 800, max_length: int = 150, min_length: int = 50) -> str:
    if not text or len(text.split()) < 60:
        # For very short text, just return it (or a trimmed version)
        return text if len(text) <= 500 else text[:500] + '...'
    summarizer = _get_summarizer()
    chunks = _chunk_text(text, max_words=max_words_chunk)
    partials = []
    for ch in chunks:
        out = summarizer(
            ch,
            max_length=max_length,
            min_length=min_length,
            do_sample=False
        )[0]['summary_text']
        partials.append(out)
    # If multiple partials, summarize the summaries once more
    if len(partials) > 1:
        joined = " ".join(partials)
        final = summarizer(joined, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
        return final
    return partials[0]