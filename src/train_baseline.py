import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
import joblib
from preprocess import clean_text

def load_data(path):
    df = pd.read_csv(path)
    # Expect columns: text, label (1=fake, 0=real). Adapt here if needed.
    assert 'text' in df.columns and 'label' in df.columns, "CSV must have 'text' and 'label' columns."
    df = df.dropna(subset=['text', 'label'])
    df['text'] = df['text'].astype(str).map(clean_text)
    return df

def build_pipeline():
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=50000, ngram_range=(1,2), strip_accents='unicode')),
        ('clf', LogisticRegression(max_iter=2000, solver='saga', n_jobs=None, class_weight='balanced'))
    ])
    return pipe

def main(args):
    df = load_data(args.data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    if hasattr(pipe, 'predict_proba'):
        proba = pipe.predict_proba(X_test)[:,1]
        auc = roc_auc_score(y_test, proba)
    else:
        auc = None
    print(classification_report(y_test, preds))
    if auc is not None:
        print(f"ROC-AUC: {auc:.4f}")
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    joblib.dump(pipe, args.model_path)
    print(f"Saved model to {args.model_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, default='models/baseline.joblib')
    args = parser.parse_args()
    main(args)