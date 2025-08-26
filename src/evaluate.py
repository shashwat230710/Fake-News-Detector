import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import joblib
from preprocess import clean_text

def main(args):
    df = pd.read_csv(args.data_path)
    df = df.dropna(subset=['text','label'])
    df['text'] = df['text'].astype(str).map(clean_text)
    model = joblib.load(args.model_path)
    X = df['text']
    y = df['label']
    y_pred = model.predict(X)
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred),
    }
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X)[:,1]
        try:
            metrics['roc_auc'] = roc_auc_score(y, y_prob)
        except Exception:
            pass
    for k,v in metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, default='models/baseline.joblib')
    args = parser.parse_args()
    main(args)