import argparse
import joblib
from .preprocess import clean_text
from .summarizer import summarize_text
from .scraper import get_article_text

def load_model(model_path='models/baseline.joblib'):
    return joblib.load(model_path)

def classify_text(text, model):
    text_clean = clean_text(text)
    proba = None
    if hasattr(model, 'predict_proba'):
        proba = float(model.predict_proba([text_clean])[:,1][0])
    pred = int(model.predict([text_clean])[0])
    return pred, proba

def analyze(text=None, url=None, model_path='models/baseline.joblib', do_summary=True):
    model = load_model(model_path)
    if url and not text:
        text = get_article_text(url)
    if not text or len(text.strip()) == 0:
        return {'error': 'No text provided or extracted.'}
    label, prob = classify_text(text, model)
    result = {
        'label': 'FAKE' if label==1 else 'REAL',
        'probability_fake': prob,
    }
    if do_summary:
        try:
            result['summary'] = summarize_text(text)
        except Exception as e:
            result['summary_error'] = str(e)
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, default=None)
    parser.add_argument('--url', type=str, default=None)
    parser.add_argument('--model_path', type=str, default='models/baseline.joblib')
    args = parser.parse_args()

    if not args.text and not args.url:
        print("Provide --text or --url")
    else:
        out = analyze(text=args.text, url=args.url, model_path=args.model_path, do_summary=True)
        print(out)